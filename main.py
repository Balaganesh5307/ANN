"""
Project Title: Development of an Artificial Neural Network (ANN) for Binary Classification 
Using the make_moons Dataset from Scikit-learn.

Description:
Modular, clean, and production-ready machine learning pipeline that builds, trains,
evaluates, and visualizes an Artificial Neural Network (ANN) on non-linearly
separable data from Scikit-learn's make_moons dataset.
"""

import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Ensure prompt output is flushed immediately to terminal
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Global Constants & Hyperparameters
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
N_SAMPLES: int = 500
DATASET_NOISE: float = 0.2
TEST_SIZE: float = 0.2
VAL_SPLIT: float = 0.2

LEARNING_RATE: float = 0.005
BATCH_SIZE: int = 32
MAX_EPOCHS: int = 150
EARLY_STOPPING_PATIENCE: int = 20

BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PLOT_PATH: str = os.path.join(BASE_DIR, "results_visualization.png")


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for NumPy and TensorFlow to ensure reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def generate_and_split_data(
    n_samples: int = N_SAMPLES,
    noise: float = DATASET_NOISE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic non-linear 2D moon-shaped dataset and split into train/test sets.

    Returns:
        Tuple containing (X, y, X_train, X_test, y_train, y_test).
    """
    print("\n[Step 1] Generating make_moons dataset...")
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    print(f"Total samples generated: {X.shape[0]}")
    print(f"Feature shape: {X.shape}, Label shape: {y.shape}")
    print(f"Classes: {np.unique(y)} (Binary classification: 0 and 1)")

    print(f"\n[Step 2] Splitting data into training ({(1 - test_size) * 100:.0f}%) and testing ({test_size * 100:.0f}%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples:  {X_test.shape[0]}")

    return X, y, X_train, X_test, y_train, y_test


def build_ann_model(learning_rate: float = LEARNING_RATE) -> tf.keras.Model:
    """
    Construct and compile the Artificial Neural Network.

    Architecture:
        - Input: 2 features (X1, X2)
        - Hidden Layer 1: 32 units, ReLU activation
        - Hidden Layer 2: 16 units, ReLU activation
        - Output Layer: 1 unit, Sigmoid activation (calibrated binary probability)
    """
    print("\n[Step 3] Building the Neural Network architecture...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(32, activation="relu", name="hidden_layer_1"),
        tf.keras.layers.Dense(16, activation="relu", name="hidden_layer_2"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="output_layer"),
    ], name="Moon_Binary_Classifier_ANN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def train_ann_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    val_split: float = VAL_SPLIT,
    patience: int = EARLY_STOPPING_PATIENCE,
) -> tf.keras.callbacks.History:
    """Train the model with validation monitoring and early stopping."""
    print("\n[Step 4] Training the ANN with validation monitoring...")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1,
    )
    print("Training completed successfully.")
    return history


def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, float]:
    """Evaluate trained model on the unseen test dataset."""
    print("\n[Step 5] Evaluating the model on test data...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    return test_loss, test_accuracy


def print_sample_predictions(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_samples: int = 10,
) -> None:
    """Print tabular comparison of predictions versus ground truth for sample test items."""
    print(f"\n--- Sample Predictions (First {num_samples} Test Samples) ---")
    raw_predictions = model.predict(X_test[:num_samples], verbose=0)
    predicted_classes = (raw_predictions >= 0.5).astype(int).flatten()

    header = f"{'Sample':<8}{'Actual Class':<15}{'Predicted Prob':<18}{'Predicted Class':<18}{'Status'}"
    print(header)
    print("-" * len(header))

    for i in range(num_samples):
        actual = y_test[i]
        prob = raw_predictions[i][0]
        pred = predicted_classes[i]
        status = "Correct" if actual == pred else "Incorrect"
        print(f"{i + 1:<8}{actual:<15}{prob:<18.4f}{pred:<18}{status}")


def plot_results(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: tf.keras.Model,
    history: tf.keras.callbacks.History,
    save_path: str = OUTPUT_PLOT_PATH,
) -> None:
    """Create a 3-panel visualization: dataset, training curves, and learned decision boundary."""
    print("\n[Step 6] Generating visualization graphs...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Dataset Distribution
    ax1 = axes[0]
    ax1.scatter(X[y == 0, 0], X[y == 0, 1], color="#1f77b4", label="Class 0", alpha=0.8, edgecolors="k")
    ax1.scatter(X[y == 1, 0], X[y == 1, 1], color="#ff7f0e", label="Class 1", alpha=0.8, edgecolors="k")
    ax1.set_title("1. make_moons Dataset Visualization", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Feature 1 (X1)")
    ax1.set_ylabel("Feature 2 (X2)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Training & Validation Curves
    ax2 = axes[1]
    epochs_range = range(1, len(history.history["accuracy"]) + 1)
    ax2.plot(epochs_range, history.history["loss"], label="Train Loss", color="#d62728", linewidth=2)
    ax2.plot(epochs_range, history.history["val_loss"], label="Val Loss", color="#ff9896", linewidth=2, linestyle="--")
    ax2.plot(epochs_range, history.history["accuracy"], label="Train Accuracy", color="#2ca02c", linewidth=2)
    ax2.plot(epochs_range, history.history["val_accuracy"], label="Val Accuracy", color="#98df8a", linewidth=2, linestyle="--")
    ax2.set_title("2. Training & Validation Performance", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Metric Value")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: ANN Decision Boundary
    ax3 = axes[2]
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_preds = model.predict(grid_points, verbose=0)
    zz = (grid_preds >= 0.5).astype(int).reshape(xx.shape)

    ax3.contourf(xx, yy, zz, alpha=0.3, cmap=plt.cm.coolwarm)
    ax3.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color="#1f77b4", label="Class 0 (Test)", edgecolors="k")
    ax3.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color="#ff7f0e", label="Class 1 (Test)", edgecolors="k")
    ax3.set_title("3. ANN Decision Boundary", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Feature 1 (X1)")
    ax3.set_ylabel("Feature 2 (X2)")
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved as '{os.path.basename(save_path)}'.")

    # Display plot without hanging execution unless --show is explicitly passed
    if "--show" in sys.argv:
        print("Displaying graphs (close the plot window to finish)...")
        plt.show()
    else:
        plt.show(block=False)
        plt.pause(1)
        plt.close()


def main() -> None:
    """Main execution pipeline."""
    set_seed(RANDOM_SEED)

    print("=" * 65)
    print("  Artificial Neural Network (ANN) - Binary Classification")
    print("  Dataset: Scikit-learn make_moons (Non-linear Classification)")
    print("=" * 65)

    # Step 1 & 2: Data Generation and Train/Test Split
    X, y, X_train, X_test, y_train, y_test = generate_and_split_data()

    # Step 3: Model Architecture
    model = build_ann_model()

    # Step 4: Model Training
    history = train_ann_model(model, X_train, y_train)

    # Step 5: Evaluation & Sample Predictions
    evaluate_model(model, X_test, y_test)
    print_sample_predictions(model, X_test, y_test)

    # Step 6: Visualization
    plot_results(X, y, X_test, y_test, model, history)

    print("\nProject execution finished successfully.")


if __name__ == "__main__":
    main()
