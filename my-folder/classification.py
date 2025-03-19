import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load the CSV file, assuming it has a header
embedding_name = "manual_features"
file_path = '/home/selim2022/Bitirme/manual_features.csv'

embedding_name = "manual_features"
file_path = f'/home/selim2022/Bitirme/model encoding save/{embedding_name}.csv'

embedding_name = "word2vec_25000"
file_path = "Encode_Word2Vec_25000_lines.csv"


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

# Optionally, check for non-binary labels
non_binary_ids = [id_list[i] for i in range(len(y)) if y[i] not in {0, 1}]
if non_binary_ids:
    print("Non-binary labels found for the following IDs:")
    print(non_binary_ids)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the classifiers to evaluate
classifiers = {
    "Random Forest": RandomForestClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes (Gaussian)": GaussianNB(),
    "SVM": SVC(),
    "Multi-Layer Perceptron (MLP)": MLPClassifier(max_iter=1000),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
    "AdaBoost": AdaBoostClassifier()
}
'''
# Define the classifiers to evaluate
classifiers = {
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes (Gaussian)": GaussianNB(),
    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis()
}
'''

# Dictionary to store results for each classifier
results = {}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    print(f"Training {name}...")
    
    start_time = time.time()  # Start the timer
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    end_time = time.time()  # End the timer
    
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        "classifier_name": name,
        "precision_0": report["0"]["precision"],
        "recall_0": report["0"]["recall"],
        "f1_0": report["0"]["f1-score"],
        "precision_1": report["1"]["precision"],
        "recall_1": report["1"]["recall"],
        "f1_1": report["1"]["f1-score"],
        "accuracy": accuracy,
        "elapsed_time": elapsed_time  # Save the elapsed time
    }

# Convert results to a DataFrame for easy analysis
results_df = pd.DataFrame(results).T

# Create a DataFrame for each metric's ranking
precision_0_ranked = results_df[['precision_0']].sort_values(by='precision_0', ascending=False)
recall_0_ranked = results_df[['recall_0']].sort_values(by='recall_0', ascending=False)
f1_0_ranked = results_df[['f1_0']].sort_values(by='f1_0', ascending=False)

precision_1_ranked = results_df[['precision_1']].sort_values(by='precision_1', ascending=False)
recall_1_ranked = results_df[['recall_1']].sort_values(by='recall_1', ascending=False)
f1_1_ranked = results_df[['f1_1']].sort_values(by='f1_1', ascending=False)

accuracy_ranked = results_df[['accuracy']].sort_values(by='accuracy', ascending=False)

# Combine all rankings into a single summary DataFrame
summary_df = pd.DataFrame({
    "Precision Class 0": precision_0_ranked.index.tolist(),
    "Recall Class 0": recall_0_ranked.index.tolist(),
    "F1 Score Class 0": f1_0_ranked.index.tolist(),
    "Precision Class 1": precision_1_ranked.index.tolist(),
    "Recall Class 1": recall_1_ranked.index.tolist(),
    "F1 Score Class 1": f1_1_ranked.index.tolist(),
    "Accuracy": accuracy_ranked.index.tolist(),
})

# Print the summary of rankings
print("\n--- Classifier Rankings by Metric ---")
print(summary_df)

# Save the summary to a CSV file
summary_df.to_csv(f"classifier_rankings_summary_{embedding_name}.csv", index=False)
print("\nClassifier rankings have been saved to 'classifier_rankings_summary.csv'")


# Save all results to a file
results_df.to_csv(f"detailed_classification_results_{embedding_name}.csv", index=False)
print("\nAll results have been saved to 'detailed_classification_results.csv'")