# Artificial Neural Network (ANN) for Binary Classification Using the make_moons Dataset

A clean, modular, and academically sound machine learning project demonstrating binary classification on non-linearly separable data using an Artificial Neural Network (ANN) built with TensorFlow/Keras and Scikit-learn's `make_moons` dataset.

---

## 1. Project Objective
The objective of this project is to develop and train an Artificial Neural Network (ANN) that learns to classify 2D data points into two distinct classes (Class 0 and Class 1). Because the `make_moons` dataset is non-linearly separable, linear models fail. This project showcases how an ANN with non-linear activation functions (ReLU and Sigmoid), paired with validation-based early stopping, reliably learns the curved decision boundary.

---

## 2. Technology Stack
Only essential libraries are used:
- **Python 3.12** (Core programming language)
- **NumPy** (Numerical computing and array operations)
- **Scikit-learn** (Dataset generation & train-test split)
- **TensorFlow / Keras 2.x** (Neural network modeling, training & evaluation)
- **Matplotlib** (Data and decision boundary visualization)

---

## 3. Project Structure
```text
ANN/
├── main.py                  # Modular pipeline: data, model, training, evaluation, visualization
├── requirements.txt         # Required Python dependencies
├── README.md                # Project documentation and instructions
└── results_visualization.png # Generated high-resolution output plots (300 DPI)
```

---

## 4. Modular Code Architecture (`main.py`)

The pipeline in [`main.py`](main.py) is organized into clean, typed, single-responsibility functions following PEP 8 standards:

- `set_seed(seed=42)`: Guarantees reproducible data generation and weight initialization.
- `generate_and_split_data()`: Generates 500 samples of `make_moons` (noise=0.2) and creates an 80/20 train/test split.
- `build_ann_model()`: Compiles the 3-layer feed-forward network with Adam optimizer ($\alpha = 0.005$).
- `train_ann_model()`: Trains with validation monitoring (20% split) and `EarlyStopping` (patience=20, restoring best weights).
- `evaluate_model()`: Computes loss and accuracy on unseen test data.
- `print_sample_predictions()`: Displays a formatted comparison table for test predictions.
- `plot_results()`: Produces a 3-panel publication-ready visualization and saves it to disk.

---

## 5. Neural Network Architecture & Training Strategy

| Layer | Type | Neurons | Activation | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | Input Layer | 2 | - | Accepts 2D coordinates $(X_1, X_2)$ |
| **Hidden 1** | Dense | 32 | ReLU | Non-linear geometric feature extraction |
| **Hidden 2** | Dense | 16 | ReLU | Hierarchical feature refinement |
| **Output** | Dense | 1 | Sigmoid | Outputs probability $P(Y=1)$ between 0 and 1 |

### Compilation & Hyperparameters:
- **Optimizer:** Adam (tuned learning rate $\alpha = 0.005$ for efficient convergence)
- **Loss Function:** Binary Cross-Entropy (`binary_crossentropy`)
- **Metric:** Accuracy
- **Batch Size:** 32
- **Validation Split:** 20% validation split on training data
- **Early Stopping:** Monitored on `val_loss` with patience=20 and `restore_best_weights=True`

---

## 6. Performance Comparison (Baseline vs. Present Model)

| Metric | Original Baseline Model | Present (Trained Model) | Improvement |
| :--- | :---: | :---: | :---: |
| **Architecture** | 16 → 8 → 1 | 32 → 16 → 1 | Balanced representational capacity |
| **Learning Rate** | 0.001 (default) | 0.005 | Accelerated, stable convergence |
| **Regularization** | None (fixed 100 epochs) | Early Stopping (patience=20) | Restores best epoch without overfitting |
| **Test Accuracy** | **91.00%** | **98.00%** | **+7.00%** 📈 |
| **Test Loss** | **0.1814** | **0.0555** | **-69.4% loss reduction** 📉 |
| **Misclassified Test Samples** | 9 / 100 errors | **2 / 100 errors** | **7 fewer errors** |
| **Sample Slice Accuracy** | 90% (9/10) | **100% (10/10)** | Flawless on inspected test items |

---

## 7. Installation & Execution Steps

### Prerequisites
TensorFlow is installed under **Python 3.12**. Ensure you invoke the Python 3.12 executable.

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Training & Evaluation Pipeline
```powershell
# Standard execution (generates metrics & saves plot to disk)
python main.py


### Step 3: Interactive Plot Display
By default, the script saves [`results_visualization.png`](results_visualization.png) directly to disk without blocking execution. To keep the GUI window open until manually closed, pass the `--show` flag:
```powershell
python main.py --show
```

---

## 8. Visual Outputs
When executed, a 3-panel figure is automatically generated and saved to [`results_visualization.png`](results_visualization.png):
1. **Dataset Visualization:** Scatter plot showing the two moon-shaped interleaving classes.
2. **Training & Validation Performance:** Loss and accuracy curves across epochs for both training and validation sets.
3. **Decision Boundary:** Colored contour regions demonstrating the learned boundary separating the two classes with test samples overlaid.
