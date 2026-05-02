# ML From Scratch

Implementations of core machine learning algorithms from scratch using NumPy, validated against scikit-learn. The goal is not to build production models — it is to understand what happens inside `.fit()` and `.predict()` by implementing both manually and verifying against a trusted reference.

---

## Algorithms Implemented

### 1. Linear Regression — California Housing Dataset

**File:** `linear_regression_scratch/script/linear_regression.py`

Custom `LinearRegression` class trained with **batch gradient descent** on the [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) (20,640 samples, 9 features).

**Gradient update rule:**
```
w := w − α · (1/m) · Xᵀ · (ŷ − y)
b := b − α · (1/m) · Σ(ŷ − y)
```

**Results (80/20 split, learning_rate=0.01, n_iters=1000):**

| Metric | From Scratch | scikit-learn | Baseline (mean) |
|--------|-------------|--------------|-----------------|
| MAE    | $50,899.53  | $50,670.49   | $90,606.85      |
| MSE    | 4,982,028,932 | 4,908,290,571 | 13,106,960,720 |
| R²     | 0.6198      | 0.6254       | -0.0002         |

The scratch model is within ~1–5% of sklearn per sample. The small gap is expected: gradient descent converges iteratively while sklearn uses the exact closed-form normal equation.

---

### 2. Logistic Regression — Customer Churn Dataset

**File:** `logistic_regression_scratch/script/logisticregression.py`

Custom `LogisticRegression` class trained with **batch gradient descent** and **binary cross-entropy loss** on a customer churn dataset (7,043 samples, 19 features).

**Sigmoid + prediction:**
```
ŷ = σ(Xw + b) = 1 / (1 + e^(−z))
class = 1 if ŷ > 0.5 else 0
```

**Loss function (binary cross-entropy):**
```
J = −(1/m) · Σ [ y·log(ŷ) + (1−y)·log(1−ŷ) ]
```

**Results:** Predictions match scikit-learn's `LogisticRegression` exactly on test data.

---

## Project Structure

```
ML_scratch/
├── linear_regression_scratch/
│   ├── script/
│   │   ├── linear_regression.py    # Custom LinearRegression class
│   │   └── data_loader.py          # Loads California Housing CSV
│   └── notebooks/
│       └── model_eval.ipynb        # Training, preprocessing, comparison
│
└── logistic_regression_scratch/
    ├── script/
    │   ├── logisticregression.py   # Custom LogisticRegression class
    │   └── load_data.py            # Loads customer churn CSV
    └── notebooks/
        └── model_eval_logistic.ipynb  # Training, preprocessing, comparison
```

---

## Key Concepts

### Vectorized Gradient Computation

No Python loops over samples. All gradients are computed in a single matrix operation:

```python
# shape: (m,) → (n,) — sums across all samples at once
dw = (1/m) * np.dot(X.T, (y_pred - y))
db = (1/m) * np.sum(y_pred - y)
```

### Preprocessing Pipeline (No Data Leakage)

All imputation and scaling statistics are fit **only on training data** and applied to test data:

```python
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("encoder", OneHotEncoder())
])
```

### Numerical Stability (Logistic Regression)

Probabilities are clipped to `[1e-15, 1-1e-15]` before computing log to avoid `log(0)`:

```python
y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
```

---

## Debugging Story

The linear regression model initially returned arrays full of `NaN`. Differential testing — feeding the same data to sklearn's `LinearRegression` — raised a clear `ValueError: Input X contains NaN`, which confirmed the bug was in the pipeline, not the model.

**Root cause:** The California Housing dataset has 207 missing values in `total_bedrooms`. The pipeline was passing raw NaN values directly into `StandardScaler`.

**Fix:** Added `SimpleImputer(strategy="median")` before `StandardScaler` in the numeric sub-pipeline.

---

## Usage

```python
from script.linear_regression import LinearRegression

model = LinearRegression(learning_rate=0.01, n_iters=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Plot convergence
import matplotlib.pyplot as plt
plt.plot(model.loss_history)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Gradient Descent Convergence")
plt.show()
```

```python
from script.logisticregression import LogisticRegression

model = LogisticRegression(learning_rate=0.01, n_iters=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
proba = model.predict_proba(X_test)
```

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| Python 3.13 | Language |
| NumPy | All math (vectorized) |
| pandas | Data loading and manipulation |
| scikit-learn | Preprocessing pipelines, reference models, metrics |
| matplotlib | Loss curve visualization |

---

## What's Next

- Decision Trees from scratch
- K-Nearest Neighbors
- Neural network (single hidden layer) with backpropagation
