from google.colab import drive
drive.mount('/content/drive')
folder_path = '/content/drive/MyDrive/DS230_Project'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
orders = pd.read_csv(os.path.join(folder_path, "orders.csv"))
prior = pd.read_csv(os.path.join(folder_path, "order_products__prior.csv"))
train = pd.read_csv(os.path.join(folder_path, "order_products__train.csv"))
products = pd.read_csv(os.path.join(folder_path, "products.csv"))
aisles = pd.read_csv(os.path.join(folder_path, "aisles.csv"))
departments = pd.read_csv(os.path.join(folder_path, "departments.csv"))
#aisles و departments
products_full = products.merge(aisles, on="aisle_id", how="left").merge(departments, on="department_id", how="left")
print("products_full:", products_full.shape)

# prior و train مع products_full
prior_full = prior.merge(products_full, on="product_id", how="left")
train_full = train.merge(products_full, on="product_id", how="left")
print("prior_full (after products):", prior_full.shape)
print("train_full (after products):", train_full.shape)

# prior_full و train_full مع orders
prior_full = prior_full.merge(orders, on="order_id", how="left")
train_full = train_full.merge(orders, on="order_id", how="left")
print("prior_full (after orders):", prior_full.shape)
print("train_full (after orders):", train_full.shape)

user_orders = prior_full.groupby("user_id")["order_id"].nunique()
user_orders.describe()

product_orders = prior_full.groupby("product_id")["order_id"].count()
product_orders.describe()

prior_full["days_since_prior_order"].isna().mean()
prior_full["days_since_prior_order"] = prior_full["days_since_prior_order"].fillna(0)

int_cols = [
    "user_id", "order_id", "product_id",
    "order_number", "add_to_cart_order",
    "reordered", "order_dow", "order_hour_of_day"
]

for col in int_cols:
    if col in prior_full.columns:
        prior_full[col] = pd.to_numeric(prior_full[col], downcast="integer")
    if col in train_full.columns:
        train_full[col] = pd.to_numeric(train_full[col], downcast="integer")

prior_full.memory_usage(deep=True).sum() / 1024**2
train_full.memory_usage(deep=True).sum() / 1024**2


prior_full = prior_full[(prior_full["order_hour_of_day"] >= 0) & (prior_full["order_hour_of_day"] <= 23)]
prior_full = prior_full[prior_full["days_since_prior_order"] >= 0]



missing_percent = prior_full.isna().mean() * 70
missing_percent = missing_percent[missing_percent > 0]
if not missing_percent.empty:
    plt.figure(figsize=(10,4))
    plt.bar(missing_percent.index, missing_percent.values, color='red', alpha=0.7)
    plt.ylabel("%")
    plt.title("Column")
    plt.xticks(rotation=45)
    plt.show()

numeric_cols = ["days_since_prior_order", "add_to_cart_order", "order_number"]
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    plt.hist(prior_full[col], bins=15, density=True, alpha=0.5, color='black')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.show()


categorical_cols = ["order_dow", "order_hour_of_day"]
for col in categorical_cols:
    counts = prior_full[col].value_counts().sort_index()
    plt.figure(figsize=(4,4))
    plt.bar(counts.index, counts.values, color='red', alpha=0.6)
    plt.title(f"Counts of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()
