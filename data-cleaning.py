import pandas as pd
import matplotlib.pyplot as plt

# Create Dataset Directly
data = {
    "ID": [1,2,3,4,5,6,7,8,9,10],
    "Product": ["Laptop","Mobile","Headphones","Laptop","Tablet",
                "Mobile","Monitor","Keyboard","Mouse","Tablet"],
    "Price": [50000,20000,None,50000,15000,
              20000,100000,1500,800,15000],
    "Quantity": [5,10,15,5,8,10,2,20,25,8],
    "Rating": [4.5,4.2,4.0,4.5,3.8,4.2,4.7,None,4.1,3.8]
}

df = pd.DataFrame(data)

print("Original Data")
print(df)

# 1. Fill Missing Values
df["Price"] = df["Price"].fillna(df["Price"].mean())
df["Rating"] = df["Rating"].fillna(df["Rating"].mean())

# 2. Remove Duplicates
df = df.drop_duplicates()

# 3. Detect Outliers using IQR
Q1 = df["Price"].quantile(0.25)
Q3 = df["Price"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df["Price"] < lower) | (df["Price"] > upper)]

print("\nOutliers:")
print(outliers)

# 4. Statistical Measures
print("\nStatistical Measures")
print(df.describe())

# 5. Histogram
plt.figure(figsize=(6,4))
plt.hist(df["Price"], bins=5)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# 6. Bar Plot
product_sales = df.groupby("Product")["Quantity"].sum()

plt.figure(figsize=(6,4))
product_sales.plot(kind="bar")
plt.title("Product Sales")
plt.xlabel("Product")
plt.ylabel("Quantity")
plt.show()