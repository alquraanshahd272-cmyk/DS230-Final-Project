import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

aisles = pd.read_csv("aisles (1).csv")
departments = pd.read_csv("departments (1).csv")

orders = pd.read_csv("orders.csv (1).zip")
order_products_prior = pd.read_csv("order_products__prior.csv (1).zip")
order_products_train = pd.read_csv("order_products__train.csv (1).zip")
products = pd.read_csv("products.csv (2).zip")

products_full = (
    products
    .merge(aisles, on="aisle_id", how="left")
    .merge(departments, on="department_id", how="left")
)
print("products_full:", products_full.shape)


prior_full = order_products_prior.merge(products_full, on="product_id", how="left")
print("prior_full:", prior_full.shape)


train_full = order_products_train.merge(products_full, on="product_id", how="left")
print("train_full:", train_full.shape)


prior_full = prior_full.merge(orders, on="order_id", how="left")
train_full = train_full.merge(orders, on="order_id", how="left")

print("prior_full (after orders):", prior_full.shape)
print("train_full (after orders):", train_full.shape)

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

numeric_cols = [ "add_to_cart_order","days_since_prior_order", "order_number"]
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


corr = prior_full[numeric_cols].corr()
plt.figure(figsize=(5,4))
plt.imshow(corr, cmap='Blues', interpolation='none', aspect='auto')
plt.xticks(range(len(numeric_cols)), numeric_cols,fontsize=5)
plt.yticks(range(len(numeric_cols)), numeric_cols)

plt.title("Correlation Matrix")
plt.show()



hour_counts = prior_full["order_hour_of_day"].value_counts().sort_index()
plt.figure(figsize=(6,4))
plt.bar(hour_counts.index, hour_counts.values, color='black')
plt.title("Orders by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Count")
plt.show()


dow_counts = prior_full["order_dow"].value_counts().sort_index()
plt.figure(figsize=(6,4))
plt.bar(dow_counts.index, dow_counts.values, color='black')
plt.title("Orders by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Count")
plt.show()

