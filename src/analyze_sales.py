
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "ecommerce_sales_raw.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load
df = pd.read_csv(DATA_PATH)

print("Raw shape:", df.shape)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# Cleaning
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Unit Price"] = pd.to_numeric(df["Unit Price"], errors="coerce")

df = df.drop_duplicates().copy()

# Invalid quantity/price are treated as missing
df.loc[df["Quantity"] <= 0, "Quantity"] = np.nan
df.loc[df["Unit Price"] <= 0, "Unit Price"] = np.nan

# Fill categorical missing values
df["City"] = df["City"].fillna("Unknown")
df["Payment Method"] = df["Payment Method"].fillna("Unknown")
df["Customer ID"] = df["Customer ID"].fillna("Unknown")

# Remove rows where core numeric fields cannot be used
df = df.dropna(subset=["Order Date", "Quantity", "Unit Price"]).copy()

# Feature engineering
df["Revenue"] = df["Quantity"] * df["Unit Price"]
df["Month"] = df["Order Date"].dt.to_period("M").astype(str)
df["Month Name"] = df["Order Date"].dt.strftime("%b")
df["Year"] = df["Order Date"].dt.year

# KPIs
total_revenue = df["Revenue"].sum()
total_orders = df["Order ID"].nunique()
total_quantity = df["Quantity"].sum()
avg_order_value = df.groupby("Order ID")["Revenue"].sum().mean()
unique_customers = df["Customer ID"].nunique()

print("\n=== KEY METRICS ===")
print(f"Total Revenue: ₹{total_revenue:,.2f}")
print(f"Total Orders: {total_orders:,}")
print(f"Total Quantity Sold: {total_quantity:,.0f}")
print(f"Average Order Value: ₹{avg_order_value:,.2f}")
print(f"Unique Customers: {unique_customers:,}")

# Aggregations
monthly = df.groupby("Month", as_index=False)["Revenue"].sum()
category = df.groupby("Category", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
region = df.groupby("Region", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
product = df.groupby("Product", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False).head(10)
payment = df.groupby("Payment Method", as_index=False)["Order ID"].nunique().sort_values("Order ID", ascending=False)
city = df.groupby("City", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False).head(10)

# Save cleaned data
clean_path = os.path.join(BASE_DIR, "data", "ecommerce_sales_cleaned.csv")
df.to_csv(clean_path, index=False)

# Plot helper

plt.figure(figsize=(10,5))
plt.plot(monthly["Month"], monthly["Revenue"], marker="o")
plt.title("Monthly Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue (₹)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "monthly_revenue.png"), dpi=150)
plt.close()

plt.figure(figsize=(9,5))
plt.barh(category["Category"], category["Revenue"])
plt.title("Revenue by Category")
plt.xlabel("Revenue (₹)")
plt.ylabel("Category")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "revenue_by_category.png"), dpi=150)
plt.close()

plt.figure(figsize=(9,5))
plt.barh(region["Region"], region["Revenue"])
plt.title("Revenue by Region")
plt.xlabel("Revenue (₹)")
plt.ylabel("Region")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "revenue_by_region.png"), dpi=150)
plt.close()

plt.figure(figsize=(10,6))
plt.barh(product["Product"], product["Revenue"])
plt.title("Top 10 Products by Revenue")
plt.xlabel("Revenue (₹)")
plt.ylabel("Product")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_10_products.png"), dpi=150)
plt.close()

plt.figure(figsize=(8,5))
plt.barh(payment["Payment Method"], payment["Order ID"])
plt.title("Orders by Payment Method")
plt.xlabel("Number of Orders")
plt.ylabel("Payment Method")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "orders_by_payment_method.png"), dpi=150)
plt.close()

print("\n=== TOP INSIGHTS ===")
print("Top category:", category.iloc[0]["Category"])
print("Top region:", region.iloc[0]["Region"])
print("Top product:", product.iloc[0]["Product"])
print("Most-used payment method:", payment.iloc[0]["Payment Method"])

print("\nCleaned data saved to:", clean_path)
print("Charts saved to:", OUTPUT_DIR)
