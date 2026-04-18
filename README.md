Linear Regression From Scratch
A from-scratch implementation of linear regression using batch gradient descent, benchmarked against scikit-learn's LinearRegression on the California Housing dataset.
The goal of this project was not to build a production model — it was to build a deeper understanding of what actually happens inside .fit() and .predict() by implementing both manually and validating the implementation against a trusted reference.
What's Inside
script/linear_regression.py
A custom LinearRegression class with:

__init__(learning_rate, n_iters) — stores hyperparameters, initializes w and b as None (lazy initialization)
fit(X, y) — runs batch gradient descent:

Initializes w as a zero vector of length n_features, b as 0
At each iteration: predicts, computes MSE loss, computes gradients, updates parameters
Records the loss at every iteration in self.loss_history


predict(X) — applies the learned parameters: y_pred = X · w + b

Implementation is fully vectorized — no Python loops over samples. The gradient computation uses a single np.dot call that implicitly sums across samples.
script/data_loader.py
Loads the California Housing dataset.
clean_data.ipynb
End-to-end training notebook:

Loads the housing data
Engineers a rooms_per_household feature
Separates target (median_house_value) from features
Splits data with train_test_split (80/20, random_state=42)
Builds a preprocessing pipeline with ColumnTransformer:

Numeric columns: SimpleImputer(strategy="median") → StandardScaler()
Categorical columns (ocean_proximity): OneHotEncoder


Transforms train and test data
Trains the custom model and generates predictions
Trains scikit-learn's LinearRegression on the same transformed data for comparison

Key Concepts Implemented
Gradient Descent
The .fit() method performs batch gradient descent with the standard update rule:
w := w − α · dJ/dw
b := b − α · dJ/db
where α is the learning rate and the gradients of the MSE loss are:
dJ/dw = (1/m) · X.T · (ŷ − y)
dJ/db = (1/m) · Σ (ŷ − y)
The 1/(2m) constant in the loss was chosen so that the factor of 2 from the power rule cancels cleanly when taking the derivative.
Vectorization
The entire gradient computation avoids per-sample loops. For a dataset of shape (m, n):

y_pred - y → shape (m,)
np.dot(y_pred - y, X) → shape (n,) — the per-feature gradient, summed across all samples in a single operation

This is orders of magnitude faster than a Python loop and mirrors how production ML libraries operate under the hood.
Preprocessing Pipeline
All preprocessing is done inside a single sklearn Pipeline so that:

Imputation statistics (median) are fit only on training data, preventing data leakage
The scaler's mean/std are computed from training data only
Test data is transformed using the statistics learned from training
The full chain can be re-applied to any new data at prediction time with a single call

Bugs Encountered (and Fixed)
Part of what this project taught me was how to debug a model that silently produces garbage.

NaN predictions — the custom model returned arrays containing NaN. The model itself happily multiplied NaN values through without complaining.
Differential testing — fed the same transformed data to scikit-learn's LinearRegression, which raised a clear ValueError: Input X contains NaN. This confirmed the bug was in the data pipeline, not in the model.
Root cause — the total_bedrooms column in the California Housing dataset has 207 missing values. The original pipeline had StandardScaler with no imputation step, so NaN values passed straight through.
Fix — wrapped StandardScaler inside a sub-pipeline with SimpleImputer(strategy="median") placed before it.

Validation
The custom implementation was validated by comparing predictions against scikit-learn's LinearRegression on the same transformed test data. Predictions from the two models agree within roughly 1–5% per sample — the small gap is expected because gradient descent converges toward the optimum iteratively, while scikit-learn uses the exact closed-form solution (the normal equation).
Running It
pythonfrom script.linear_regression import LinearRegression

model = LinearRegression(learning_rate=0.01, n_iters=1000)
model.fit(X_train_transformed, y_train)
y_pred = model.predict(X_test_transformed)
Loss history is available as model.loss_history for plotting convergence curves.
Tech Stack

Python 3.13
NumPy (all math)
pandas (data loading, column selection)
scikit-learn (pipeline, preprocessing, reference model)
matplotlib (loss curve plotting)