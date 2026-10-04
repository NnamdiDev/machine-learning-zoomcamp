import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error


# Load dataset
df = pd.read_csv(
    "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
)

# Select relevant columns
df = df[
    [
        "engine_displacement",
        "horsepower",
        "vehicle_weight",
        "model_year",
        "fuel_efficiency_mpg",
    ]
]


# Define dataset structure
n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

features = [
    "engine_displacement",
    "horsepower",
    "vehicle_weight",
    "model_year",
]

target = "fuel_efficiency_mpg"


# Q1: Find the column with missing values
print("Missing values:")
print(df.isnull().sum())


# Q2: Find the median horsepower
print("\nMedian horsepower:")
print(df["horsepower"].median())


# Create train, validation, and test sets
np.random.seed(42)

idx = np.arange(n)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val:]]


# Prepare features and target
X_train = df_train[features].copy()
X_val = df_val[features].copy()

y_train = df_train[target].copy()
y_val = df_val[target].copy()


# Q3: Compare zero and mean imputation

# Fill missing values with 0
X_train_zero = X_train.fillna(0)
X_val_zero = X_val.fillna(0)

model = LinearRegression()
model.fit(X_train_zero, y_train)

y_pred = model.predict(X_val_zero)

rmse_zero = root_mean_squared_error(y_val, y_pred)

print("\nQ3:")
print("RMSE with 0:", round(rmse_zero, 3))


# Fill missing values with the training mean
horsepower_mean = X_train["horsepower"].mean()

X_train_mean = X_train.fillna(horsepower_mean)
X_val_mean = X_val.fillna(horsepower_mean)

model = LinearRegression()
model.fit(X_train_mean, y_train)

y_pred = model.predict(X_val_mean)

rmse_mean = root_mean_squared_error(y_val, y_pred)

print("RMSE with mean:", round(rmse_mean, 3))


# Q4: Test different Ridge regularization values
print("\nQ4:")

r_values = [0, 0.01, 0.1, 1, 5, 10, 100]

for r in r_values:
    model = Ridge(alpha=r)
    model.fit(X_train_zero, y_train)

    y_pred = model.predict(X_val_zero)

    rmse = root_mean_squared_error(y_val, y_pred)

    print(r, round(rmse, 4))


# Q5: Check the standard deviation across different seeds
scores = []

for seed in range(10):
    np.random.seed(seed)

    idx = np.arange(n)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train + n_val]]

    X_train = df_train[features].copy()
    X_val = df_val[features].copy()

    y_train = df_train[target].copy()
    y_val = df_val[target].copy()

    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    rmse = root_mean_squared_error(y_val, y_pred)

    scores.append(rmse)

print("\nQ5:")
print("Standard deviation:", round(np.std(scores), 3))


# Q6: Train final model and evaluate on test set
np.random.seed(9)

idx = np.arange(n)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val:]]

# Combine training and validation data
df_full_train = pd.concat([df_train, df_val])

X_full_train = df_full_train[features].copy()
X_test = df_test[features].copy()

y_full_train = df_full_train[target].copy()
y_test = df_test[target].copy()

# Fill missing values with 0
X_full_train = X_full_train.fillna(0)
X_test = X_test.fillna(0)

# Train final Ridge model
model = Ridge(alpha=0.001)
model.fit(X_full_train, y_full_train)

y_pred = model.predict(X_test)

rmse = root_mean_squared_error(y_test, y_pred)

print("\nQ6:")
print("Test RMSE:", round(rmse, 3))