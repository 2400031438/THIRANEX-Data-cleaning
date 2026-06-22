import pandas as pd
import matplotlib.pyplot as plt

# Sample Dataset
data = {
    "Student": ["A","B","C","D","E","F","G","H","I","J"],
    "Study_Hours": [2,3,4,5,6,7,8,3,5,9],
    "Attendance": [60,65,70,75,80,85,90,68,78,95],
    "Marks": [40,45,50,55,65,70,80,48,60,90]
}

df = pd.DataFrame(data)

print("Dataset")
print(df)

# Statistical Summary
print("\nStatistical Summary")
print(df.describe())

# Correlation
print("\nCorrelation Matrix")
print(df.corr(numeric_only=True))

# Average Marks
print("\nAverage Marks:", df["Marks"].mean())

# Bar Chart
plt.figure(figsize=(6,4))
plt.bar(df["Student"], df["Marks"])
plt.title("Student vs Marks")
plt.xlabel("Student")
plt.ylabel("Marks")
plt.show()

# Scatter Plot
plt.figure(figsize=(6,4))
plt.scatter(df["Study_Hours"], df["Marks"])
plt.title("Study Hours vs Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()

# Histogram
plt.figure(figsize=(6,4))
plt.hist(df["Marks"], bins=5)
plt.title("Marks Distribution")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.show()