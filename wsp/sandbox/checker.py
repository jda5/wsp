"""
Decides whether a cell of agent code may run, before any of it runs.

The rule is "allow what is listed", never "ban what is known to be bad":
every node in the cell's syntax tree must be one of ALLOWED_NODES, and no
name or attribute may start with an underscore or appear in
FORBIDDEN_ATTRIBUTES. Anything else rejects the whole cell.

What this rules out, and why:

    import, from ... import    numpy is already loaded as `np`
    for, while, comprehensions no loops, so no training loops
    def, lambda, class         no new behaviour, only numpy calls
    try, with, global, del     not needed to edit weights
    x.attr = ...               would change numpy objects the harness shares
    _private, __dunder__       closes the classic escapes such as
                               ().__class__.__bases__[0].__subclasses__()
"""

import ast


class ForbiddenCode(Exception):
    """
    The cell uses something that is not on the allow-list.
    """


ALLOWED_NODES = (
    # statements
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.AugAssign,
    # names, values and indexing: W["dense.kernel"][3:10, 0]
    ast.Name,
    ast.Constant,
    ast.Subscript,
    ast.Slice,
    ast.Tuple,
    ast.List,
    ast.Dict,
    ast.Starred,
    # calls and attributes: np.outer(u, v), W[k].T
    ast.Call,
    ast.keyword,
    ast.Attribute,
    # arithmetic and logic
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.operator,
    ast.unaryop,
    ast.boolop,
    ast.cmpop,
    ast.expr_context,
    # f-strings, for labelled printing
    ast.JoinedStr,
    ast.FormattedValue,
)

FORBIDDEN_ATTRIBUTES = {
    # write arrays to disk
    "tofile",
    "dump",
    "dumps",
    # raw memory access
    "ctypes",
    "cffi",
    # str.format can reach attributes by name: "{0.__class__}".format(x)
    "format",
    "format_map",
    # class and frame internals
    "mro",
    "gi_frame",
    "tb_frame",
    "f_globals",
    "f_builtins",
}

HINTS = {
    ast.Import: "imports are not available; numpy is already loaded as `np`",
    ast.ImportFrom: "imports are not available; numpy is already loaded as `np`",
    ast.For: "loops are not available; use vectorised numpy instead",
    ast.While: "loops are not available; use vectorised numpy instead",
    ast.ListComp: "comprehensions are not available; use vectorised numpy instead",
    ast.DictComp: "comprehensions are not available; use vectorised numpy instead",
    ast.SetComp: "comprehensions are not available; use vectorised numpy instead",
    ast.GeneratorExp: "comprehensions are not available; use vectorised numpy instead",
    ast.FunctionDef: "defining functions is not available",
    ast.Lambda: "lambdas are not available",
}


def check(code: str) -> ast.Module:
    """
    Parse ``code`` and return its syntax tree, or raise ``ForbiddenCode``.
    """
    tree = ast.parse(code)

    for node in ast.walk(tree):
        _check_node(node)

    return tree


def _check_node(node: ast.AST) -> None:
    line = getattr(node, "lineno", "?")

    if not isinstance(node, ALLOWED_NODES):
        hint = HINTS.get(type(node), f"`{type(node).__name__}` is not available")
        raise ForbiddenCode(f"Line {line}: {hint}.")

    if isinstance(node, ast.Name) and node.id.startswith("_"):
        raise ForbiddenCode(f"Line {line}: names starting with _ are not available.")

    if isinstance(node, ast.Attribute):
        if node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRIBUTES:
            raise ForbiddenCode(f"Line {line}: `.{node.attr}` is not available.")

        # The numpy functions the agent sees are the harness's own objects, so
        # setting an attribute on one (np.mgrid.sparse = True) would change
        # numpy for the experiment too. Weights are edited by index instead.
        if isinstance(node.ctx, ast.Store):
            raise ForbiddenCode(
                f"Line {line}: assigning to attributes (`.{node.attr} = ...`) is "
                "not available; edit arrays by index, e.g. W[name][...] = ..."
            )
