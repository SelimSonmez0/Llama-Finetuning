#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 00:06:34 2024

@author: selim2022
"""

import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier


embedding_name = "word2vec_20000"
file_path = "/home/selim2022/Bitirme/document/Doc_save/Word2Vec/Encode_Word2Vec_20000_lines.csv"


data = pd.read_csv(file_path)

# Display the first few rows to understand the structure
print("Sample data:")
print(data.head())

# Extract IDs (first column), labels (last column), and encodings (middle part)
id_list = data.iloc[:, 0].tolist()  # First column as IDs
label_list = data.iloc[:, -1].tolist()  # Last column as labels
X = data.iloc[:, 1:-1].values  # Middle part as encodings (2nd column to second last column)

# Convert label_list to a suitable format (e.g., list of integers)
y = []
for label in label_list:
    try:
        y.append(int(label))
    except ValueError:
        print(f"Skipping invalid label: {label}")


# Define the LightGBM classifier
clf = LGBMClassifier()

# Train the classifier using the entire dataset
print("Training LightGBM on the entire dataset...")
start_time = time.time()
clf.fit(X, y)  # Use all data for training
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Training completed in {elapsed_time:.2f} seconds.")

# Save the trained model to a file
model_file = "lgbm_model_all_data.txt"
clf.booster_.save_model(model_file)
print(f"Model saved to {model_file}")
