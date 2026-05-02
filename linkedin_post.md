# LinkedIn Post

---

I spent the last few weeks building Linear Regression and Logistic Regression from scratch — no sklearn `.fit()`, just NumPy and math.

Here's what I actually learned:

**The algorithm is 15 lines of code.**

```python
for _ in range(self.n_iters):
    y_pred = np.dot(X, self.w) + self.b
    dw = (1/m) * np.dot(X.T, (y_pred - y))
    db = (1/m) * np.sum(y_pred - y)
    self.w -= self.lr * dw
    self.b -= self.lr * db
```

That's batch gradient descent. That's what sklearn is doing under the hood every time you call `.fit()`. The rest is engineering.

---

**What I built:**

Linear Regression trained on the California Housing dataset (20,640 samples):
- R² = 0.6198 (sklearn gets 0.6254 — the gap is expected, gradient descent vs. closed-form normal equation)
- MAE = $50,899 on median house value predictions
- Baseline (predict mean every time): R² = -0.0002

Logistic Regression trained on a customer churn dataset (7,043 samples):
- Predictions match sklearn's LogisticRegression exactly on held-out test data

---

**The most valuable part wasn't the math — it was debugging.**

My linear regression model was silently returning NaN predictions. The model happily computed gradients on NaN values without complaining.

I used differential testing: fed the same data to sklearn's `LinearRegression`, which immediately raised `ValueError: Input X contains NaN`. That told me the bug was in the data pipeline, not the model.

Root cause: the California Housing dataset has 207 missing values in `total_bedrooms`. My pipeline was feeding raw NaN into `StandardScaler`.

Fix: one line — add `SimpleImputer(strategy="median")` before the scaler.

This is the kind of thing you only internalize by breaking it yourself.

---

**Key things that clicked after doing this:**

- Why preprocessing statistics must be fit on training data only (data leakage is subtle)
- Why vectorization matters — one `np.dot` call vs. a Python loop over 20,000 samples
- Why epsilon clipping exists in logistic regression (`log(0)` will silently destroy your loss)
- Why gradient descent and the normal equation give slightly different answers
- What a preprocessing pipeline actually does at each step

---

If you're learning ML, I'd strongly recommend spending a weekend building at least one algorithm from scratch before moving on to model tuning.

You'll stop treating `.fit()` as magic.

GitHub: [link to repo]

#MachineLearning #DataScience #Python #GradientDescent #MLFromScratch #NumPy #DeepLearning
