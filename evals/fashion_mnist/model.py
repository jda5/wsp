import keras
import numpy as np

SEED = 42

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def load_data(split: float = 0.1):
    (X_train_full, y_train_full), (X_test, y_test) = (
        keras.datasets.fashion_mnist.load_data()
    )

    # Scale every split the same way (float32 in [0, 1]).
    X_train_full = X_train_full.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    N = int(split * X_train_full.shape[0])

    print(f"Using {N} samples for validation.")

    # Hold out the last split fraction of training images for validation.
    X_train, y_train = X_train_full[:-N], y_train_full[:-N]
    X_valid, y_valid = X_train_full[-N:], y_train_full[-N:]

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def build_model() -> keras.Model:
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28)),
            keras.layers.Flatten(),
            keras.layers.Dense(300, activation="relu"),
            keras.layers.Dense(100, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    keras.utils.set_random_seed(SEED)  # reproducible baseline across runs

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_data()
    model = build_model()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,  # keep the best epoch, not the last one
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=100,  # upper bound; early stopping decides when to stop
        validation_data=(X_valid, y_valid),
        callbacks=[early_stopping],
        verbose=2,  # one line per epoch
    )
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    print(
        f"Stopped after {len(history.history['val_loss'])} epochs (best: {best_epoch})"
    )

    _, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n--- Test Accuracy: {test_accuracy:.4f} ---\n")

    # Per-class accuracy shows which garments the model confuses.
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    for i, name in enumerate(CLASS_NAMES):
        mask = y_test == i
        print(f"  {name:<12} {np.mean(y_pred[mask] == i):.4f}")

    model.save("trained_fashion_mnist_model.keras")


if __name__ == "__main__":
    main()
