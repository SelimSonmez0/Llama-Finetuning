import time
import gc
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from lightgbm import Booster

# Load the pre-trained Word2Vec model
word_vectors = KeyedVectors.load_word2vec_format('trmodel', binary=True)

# Path to the saved LightGBM model
model_file = "lgbm_model_all_data.txt"

# Load the saved LightGBM model
print(f"Loading the model from {model_file}...")
model = Booster(model_file=model_file)
print("Model loaded successfully.")

# Function to compute sentence embedding
def get_sentence_embedding(sentence, word_vectors):
    sentence = str(sentence)
    words = sentence.split()
    word_embeddings = []
    for word in words:
        word_lower = word.lower()
        if word_lower in word_vectors:
            word_embeddings.append(word_vectors[word_lower])
    if not word_embeddings:
        return np.zeros(word_vectors.vector_size)
    return np.mean(word_embeddings, axis=0)

# Function to make predictions using LightGBM model
def predict_with_model(model, data):
    """
    Predict using the LightGBM model.

    Parameters:
    model (Booster): The loaded LightGBM model.
    data (ndarray or DataFrame): The input data for prediction.

    Returns:
    predictions: Predicted labels.
    probabilities: Predicted probabilities.
    """
    if isinstance(data, pd.DataFrame):
        data = data.values  # Convert DataFrame to ndarray
    probabilities = model.predict(data)  # Predicted probabilities
    predictions = np.argmax(probabilities, axis=1) if probabilities.ndim > 1 else (probabilities > 0.5).astype(int)
    return predictions, probabilities

# Step 1: Filter lines based on the random number comparison with confidence
def filter_row(probabilities):
    filtered_rows = []
    for prob in probabilities:
        random_value = np.random.rand()
        # Compare the random number with the probability
        # If random_value > probability, skip the row
        if random_value > prob:
            filtered_rows.append(True)
        else:
            filtered_rows.append(False)
    return filtered_rows

# Path to the input embeddings CSV file (example data)
input_file = "doc2line_Filtered_CulturaX_0.csv"  # Adjust the path as needed

# Read the input data from the CSV file
print(f"Reading data from {input_file}...")
data = pd.read_csv(input_file)

# Display the first few rows to verify the data
print("Sample data:")
print(data.head())

# Extract the ID column and feature columns (all columns except the first 'id' column)
ids = data.iloc[:, 0].values  # First column as IDs

# Define chunk size for processing
chunk_size = 500000  # Process data in chunks of 1000 rows (adjust as needed)

# Total number of chunks
total_chunks = (len(data) + chunk_size - 1) // chunk_size  # This ensures rounding up

# Prepare output list to store predictions
output_data = []

# Process the data in chunks to avoid memory issues
for start in range(0, len(data), chunk_size):
    end = min(start + chunk_size, len(data))
    chunk = data.iloc[start:end]
    
    # Print the chunk number and total number of chunks
    print(f"Making predictions for chunk {start}-{end} (Chunk {start // chunk_size + 1}/{total_chunks})...")
    
    # Time: Compute embeddings
    embedding_start_time = time.time()
    embeddings = chunk['Text'].apply(lambda x: get_sentence_embedding(x, word_vectors)).tolist()
    embedding_end_time = time.time()
    print(f"Embeddings for chunk {start}-{end} computed in {embedding_end_time - embedding_start_time:.2f} seconds")
    
    # Time: Make predictions
    prediction_start_time = time.time()
    predictions, probabilities = predict_with_model(model, embeddings)
    prediction_end_time = time.time()
    print(f"Predictions for chunk {start}-{end} made in {prediction_end_time - prediction_start_time:.2f} seconds")
    
    # Time: Filter rows
    filter_start_time = time.time()
    filtered_rows = filter_row(probabilities)
    filter_end_time = time.time()
    print(f"Filtering for chunk {start}-{end} completed in {filter_end_time - filter_start_time:.2f} seconds\n")
    
    # Only keep the rows that passed the filter
    filtered_chunk = chunk[filtered_rows]

    if not filtered_chunk.empty:
        # Store the results in the output chunk dataframe
        output_chunk = pd.DataFrame({
            "ID": filtered_chunk.iloc[:, 0].values,  # Include IDs in the output
            "Text": filtered_chunk['Text'],  # Include the original text
            "Prediction": predictions[filtered_rows],
            "Probability": probabilities[filtered_rows] if probabilities.ndim == 1 else probabilities[filtered_rows, 1]
        })
    
        # Append only the filtered chunk (those that passed the filter)
        output_data.append(output_chunk)
        
    # Free memory
    del chunk
    del embeddings
    del output_chunk
    gc.collect()
    
# Concatenate all the chunks' predictions into one DataFrame
final_output = pd.concat(output_data, ignore_index=True)

# Save the filtered predictions to a new CSV file
filtered_output_file = "filtered_"+input_file  # The new CSV file to save the filtered data
final_output.to_csv(filtered_output_file, index=False)

print(f"Filtered predictions saved to {filtered_output_file}")
