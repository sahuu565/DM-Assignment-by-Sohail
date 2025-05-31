
import pandas as pd
import numpy as np

try:
    train_data = pd.read_csv("train.csv")
except FileNotFoundError:
    print("Error: 'train.csv' not found. Please download it from Kaggle and place it in the working directory.")
    exit()

train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
train_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)
train_data.dropna(subset=['Cabin'], how='all', inplace=True)

age_bins = [0, 12, 18, 30, 50, 100]
age_labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
train_data['AgeGroup'] = pd.cut(train_data['Age'], bins=age_bins, labels=age_labels)

fare_bins = [0, 10, 50, 100, 1000]
fare_labels = ['Low', 'Medium', 'High', 'Very High']
train_data['FareGroup'] = pd.cut(train_data['Fare'], bins=fare_bins, labels=fare_labels)

try:
    test_data = pd.read_csv("test.csv")
    test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
    test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)
    merged_data = pd.concat([train_data, test_data], ignore_index=True)
except FileNotFoundError:
    print("Error: 'test.csv' not found. Proceeding with only train data.")
    merged_data = train_data

merged_data['Embarked'].fillna(merged_data['Embarked'].mode()[0], inplace=True)
merged_data.dropna(subset=['Fare'], inplace=True)

print("Dataset Info:")
print(merged_data.info())
print("\nFirst 5 rows:")
print(merged_data.head())

merged_data.to_csv("preprocessed_titanic.csv", index=False)
print("Preprocessed data saved to 'preprocessed_titanic.csv'")

