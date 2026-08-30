
# **EX-NO. 4(a): MACHINE LEARNING REGRESSION MODELS FOR HOUSE PRICE PREDICTION**

---

## **AIM**

To construct, train, and evaluate multiple Machine Learning regression models for predicting continuous house prices using physical and spatial attributes, and to benchmark their relative performance using standard evaluation metrics ($\text{RMSE}$, $\text{MAE}$, and $R^2$).

---

## **1. INTRODUCTION**

Machine Learning enables computational systems to recognize underlying structural patterns in historical datasets and generate predictions on unseen observations without explicit rule programming.

* **Supervised Regression:** A class of supervised learning algorithms tasked with predicting continuous quantitative target variables $y \in \mathbb{R}$ based on a feature matrix $X$.
* **Application Scope:** In real estate analytics, regression models map spatial, structural, and environmental characteristics to historical asset valuations.
* **Feature Schema:**
* `square_feet`: Living area footprint of the property (Continuous).
* `num_rooms`: Total count of habitable rooms (Discrete).
* `age`: Property age calculated in years (Continuous).
* `distance_to_city(km)`: Proximity to the central business district (Continuous).


* **Target Vector ($y$):**
* `price`: Continuous market valuation of the property.



---

## **2. DATASET DESCRIPTION**

* **Dataset Name:** House Price Dataset (`house_prices_dataset.csv`)
* **Problem Type:** Supervised Multi-Variate Continuous Regression

```
+-----------------------------------------------------------------------------------+
|                              FEATURE SCHEMATICS                                   |
+---------------------+-------------------+------------------+----------------------+
| Feature Name        | Representation    | Attribute Type   | Data Dimension       |
+---------------------+-------------------+------------------+----------------------+
| square_feet         | Input Feature (X) | Continuous       | $\mathbb{R}^+$ (sq ft)|
| num_rooms           | Input Feature (X) | Discrete         | Integer Count        |
| age                 | Input Feature (X) | Continuous       | Years                |
| distance_to_city(km)| Input Feature (X) | Continuous       | Kilometers           |
| price               | Target Label (y)  | Continuous Target| Monetary Units ($)   |
+---------------------+-------------------+------------------+----------------------+

```

---

## **3. PROBLEM STATEMENT**

Develop an end-to-end predictive machine learning pipeline to model non-linear real estate price valuations. The workflow requires loading high-dimensional data, conducting exploratory visual analysis, eliminating leverage outliers, standardizing inputs, training ten benchmark regression algorithms, and ranking model performance using standard validation metrics.

---

## **4. REGRESSION ALGORITHMS EVALUATED**

```
                     +----------------------------------+
                     |    MODEL EVALUATION SPECTRUM     |
                     +----------------------------------+
                                      |
         +----------------------------+----------------------------+
         |                                                         |
         v                                                         v
  [Linear Models]                                        [Non-Linear Models]
  - Ordinary Least Squares (OLS)                         - Polynomial Regression (Deg 2)
  - Ridge (L2 Regularization)                            - Decision Tree Regressor
  - Lasso (L1 Regularization)                            - Random Forest Ensemble
  - ElasticNet (L1 + L2 Hybrid)                          - Gradient Boosting (GBR)
                                                         - Support Vector Regressor (SVR)
                                                         - K-Nearest Neighbors (KNN)

```

---

## **5. ENVIRONMENT & LIBRARY IMPORT SCHEME**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing & Model Selection Modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Linear Models
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

# Non-Linear & Tree-Based Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Model Evaluation Metrics
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

```

---

## **6. DATASET INGESTION & STRUCTURAL PROFILING**

### **6.1 Ingestion**

```python
# Ingest local storage dataset from drive location
df = pd.read_csv('/content/drive/MyDrive/Datasets/house_prices_dataset.csv')

# Inspect top 5 observations
df.head()

```

### **6.2 Structural Inspection & Null Auditing**

```python
# Full dataframe printout
print(df)

# Structural datatype summary
df.info()

# Matrix spatial footprint (Rows, Columns)
print(f"Dataset Shape: {df.shape}")

# Descriptive statistical distribution summary
df.describe()

# Check for missing values (NaN count per feature)
print(df.isnull().sum())

```

---

## **7. EXPLORATORY DATA ANALYSIS (EDA)**

### **7.1 Feature Distribution Profiles**

Evaluating univariate structural distributions ensures feature inputs meet normality assumptions across models.

```python
numeric_features = [
    'square_feet',
    'num_rooms',
    'age',
    'distance_to_city(km)',
    'price'
]

for col in numeric_features:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Univariate Distribution: {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

```

### **7.2 Feature Correlation Matrix**

Pearson correlation coefficients ($\rho$) measure multi-collinearity across linear features:

$$\rho_{X,Y} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$$

```python
plt.figure(figsize=(8, 6))
sns.heatmap(
    df.corr(),
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=0.5
)
plt.title("Feature Correlation Matrix")
plt.show()

```

### **7.3 Feature-Target Bivariate Scatter Analysis**

```python
for col in ['square_feet', 'num_rooms', 'age', 'distance_to_city(km)']:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df[col], y=df['price'], alpha=0.7, color='teal')
    plt.title(f'{col} vs. Target Price')
    plt.xlabel(col)
    plt.ylabel('Price ($)')
    plt.show()

```

---

## **8. OUTLIER DETECTION & PERCENTILE TRIMMING**

### **8.1 Outlier Identification**

Extreme value observations inflate sum-of-squared errors ($\text{SSE}$), heavily skewing parameters in linear models.

```python
for col in numeric_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col], color='lightcoral')
    plt.title(f'Boxplot Outlier Inspection: {col}')
    plt.show()

```

### **8.2 Percentile Trimming Strategy**

Outliers are trimmed by bounding targets between the 1st ($Q_{0.01}$) and 99th ($Q_{0.99}$) percentiles:

```python
Q1 = df['price'].quantile(0.01)
Q99 = df['price'].quantile(0.99)

# Filter dataset within percentile boundaries
df = df[
    (df['price'] >= Q1) &
    (df['price'] <= Q99)
]

print(f"Cleaned Dataset Shape: {df.shape}")

```

---

## **9. PREPROCESSING & MODEL SPLITTING PIPELINE**

### **9.1 Feature-Target Separation**

```python
X = df[['square_feet', 'num_rooms', 'age', 'distance_to_city(km)']]
y = df['price']

```

### **9.2 Train-Test Data Partitioning**

The dataset is split into an **80% training set** and a **20% holdout validation set** to evaluate model generalization.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

```

### **9.3 Standard Feature Scaling**

To prevent attributes with large ranges (e.g., `square_feet`) from dominating models sensitive to feature magnitudes (e.g., SVR, KNN, Regularized Linear Models), values are scaled to zero mean and unit variance ($\mu=0, \sigma=1$):

$$z = \frac{x - \mu}{\sigma}$$

```python
scaler = StandardScaler()

# Fit scaler strictly on training features to prevent data leakage
X_train_scaled = scaler.fit_transform(X_train)

# Transform validation test features
X_test_scaled = scaler.transform(X_test)

```

---

## **10. MODEL TRAINING ARCHITECTURE**

### **10.1 Naïve Baseline Predictor**

Calculates the mean training target value across all observations to establish an uninformative baseline score:

```python
y_pred_baseline = np.mean(y_train) * np.ones_like(y_test)

rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
mae_baseline = mean_absolute_error(y_test, y_pred_baseline)

print(f"Baseline Mean Model -> RMSE: {rmse_baseline:.2f}, MAE: {mae_baseline:.2f}")

```

### **10.2 Linear Regression (OLS)**

```python
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

```

### **10.3 Ridge Regression ($L_2$ Regularization)**

Applies an $L_2$ penalty term ($\alpha \Vert{}\beta\Vert{}_2^2$) to limit coefficient magnitude:

```python
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

```

### **10.4 Lasso Regression ($L_1$ Regularization)**

Applies an $L_1$ penalty term ($\alpha \Vert{}\beta\Vert{}_1$) to enforce feature sparsity:

```python
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)

```

### **10.5 ElasticNet Regression**

Combines both $L_1$ and $L_2$ regularization penalties:

```python
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train_scaled, y_train)
y_pred_elastic = elastic.predict(X_test_scaled)

```

### **10.6 Polynomial Regression (Degree = 2)**

Generates non-linear interaction terms ($x_1^2, x_1x_2, x_2^2$):

```python
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_lr = LinearRegression()
poly_lr.fit(X_train_poly, y_train)
y_pred_poly = poly_lr.predict(X_test_poly)

```

### **10.7 Decision Tree Regressor**

```python
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)

```

### **10.8 Random Forest Regressor**

```python
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

```

### **10.9 Gradient Boosting Regressor**

```python
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gbr.fit(X_train_scaled, y_train)
y_pred_gbr = gbr.predict(X_test_scaled)

```

### **10.10 Support Vector Regressor (SVR)**

```python
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr.fit(X_train_scaled, y_train)
y_pred_svr = svr.predict(X_test_scaled)

```

### **10.11 K-Nearest Neighbors (KNN) Regressor**

```python
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

```

---

## **11. MODEL EVALUATION & BENCHMARKING**

### **11.1 Evaluation Metrics Definition**

1. **Root Mean Squared Error (RMSE):** Penalizes larger error variance heavily.

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

2. **Mean Absolute Error (MAE):** Measures average absolute deviation magnitude.

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \vert{}y_i - \hat{y}_i\vert{}$$

3. **Coefficient of Determination ($R^2$):** Quantifies the proportion of target variance explained by the model.

$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

### **11.2 Performance Metric Computation**

```python
models = {
    "Linear Regression": y_pred_lr,
    "Ridge": y_pred_ridge,
    "Lasso": y_pred_lasso,
    "ElasticNet": y_pred_elastic,
    "Polynomial Regression": y_pred_poly,
    "Decision Tree": y_pred_dt,
    "Random Forest": y_pred_rf,
    "Gradient Boosting": y_pred_gbr,
    "SVR": y_pred_svr,
    "KNN": y_pred_knn
}

results = []

for name, y_pred in models.items():
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append([name, rmse, mae, r2])

# Construct performance summary dataframe sorted by RMSE score
results_df = pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "R2"])
results_df = results_df.sort_values(by="RMSE").reset_index(drop=True)

print(results_df)

```

---

## **12. PERFORMANCE VISUALIZATION & DIAGNOSTICS**

### **12.1 Actual vs. Predicted Subplots**

Points aligning along the dashed red 45° reference line reflect accurate predictions.

```python
plt.figure(figsize=(15, 12))

for i, (name, y_pred) in enumerate(models.items()):
    plt.subplot(5, 2, i + 1)
    plt.scatter(y_test, y_pred, alpha=0.5, color='indigo')
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--',
        lw=2
    )
    plt.xlabel("Actual Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.title(f"{name}: Actual vs. Predicted")

plt.tight_layout()
plt.show()

```

### **12.2 Residual Plot Diagnostics**

Residuals represent prediction errors ($\epsilon_i = y_i - \hat{y}_i$). Unbiased models display random residual scatter centered around zero ($\epsilon = 0$).

```python
for name, y_pred in models.items():
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, color='crimson')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Predicted Price ($)")
    plt.ylabel("Residual Errors ($)")
    plt.title(f"{name}: Residual Scatter")
    plt.show()

```

---

## **13. FEATURE IMPORTANCE ANALYSIS**

Tree-based ensembles calculate feature importance scores using Mean Decrease in Impurity (MDI).

### **13.1 Random Forest Importance**

```python
importances_rf = rf.feature_importances_
feat_names = X.columns

plt.figure(figsize=(6, 4))
sns.barplot(x=importances_rf, y=feat_names, palette='viridis')
plt.title("Random Forest Feature Importance")
plt.xlabel("MDI Score")
plt.show()

```

### **13.2 Gradient Boosting Importance**

```python
importances_gbr = gbr.feature_importances_

plt.figure(figsize=(6, 4))
sns.barplot(x=importances_gbr, y=feat_names, palette='magma')
plt.title("Gradient Boosting Feature Importance")
plt.xlabel("MDI Score")
plt.show()

```

---

## **14. SUMMARY RESULTS CHART**

```python
plt.figure(figsize=(10, 6))
sns.barplot(
    x="RMSE",
    y="Model",
    data=results_df.sort_values("RMSE"),
    palette="Blues_r"
)
plt.title("RMSE Comparison Across Regression Models")
plt.xlabel("Root Mean Squared Error (Lower is Better)")
plt.ylabel("Regression Model")
plt.show()

```

---

## **CONCLUSION**

Ten regression algorithms were evaluated to predict housing valuations. By isolating top-performing algorithms using $\text{RMSE}$, $\text{MAE}$, and $R^2$ metrics, ensemble methods (Random Forest and Gradient Boosting) demonstrated higher accuracy in capturing non-linear interactions compared to standard linear baselines. Preprocessing techniques—including percentile outlier removal and standardize feature scaling—ensured overall model stability.

---
