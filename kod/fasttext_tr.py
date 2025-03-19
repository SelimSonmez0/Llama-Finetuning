import pandas as pd
import numpy as np
import fasttext
from huggingface_hub import hf_hub_download
import time 


# Specify the path to your CSV file
file_path = 'Culturax_linesWithLabels_June_df (3).csv'


# Read the CSV file, allowing for variable number of columns and no header
df = pd.read_csv(file_path, header=None, na_filter=False)

# Remove any entirely empty columns
df = df.dropna(axis=1, how='all')

# Prepare lists for id, text, and label
id_list = []
text_list = []
label_list = []

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    if index == 0:  # Skip the first row
        continue

    id_value = row[0]  # First column for ID
    label_value = None

    # Find the first non-empty label, starting from the last element
    for i in range(len(row) - 1, -1, -1):
        if row[i] != '':
            label_value = row[i]
            break

    # Check if label is binary
    if label_value not in ['0', '1']:
        print(f"Label={label_value} for ID {id_value} is not binary. Skipping this row.")
        continue  # Skip this row if the label is not binary

    # Gather all text elements between ID and label
    text_value = ','.join(row[1:-12].astype(str))
    id_list.append(id_value)
    text_list.append(text_value)
    label_list.append(label_value)

# Print a sample of the data
print("Sample data:")
for i in range(min(10, len(id_list))):  # Show 10 samples for brevity
    print(f"ID: {id_list[i]}, Text: {text_list[i]}, Label: {label_list[i]}")





# Load the FastText model from Hugging Face Hub
model_name = "fasttext-tr"
model_path = hf_hub_download(repo_id="facebook/fasttext-tr-vectors", filename="model.bin")
model = fasttext.load_model(model_path)

# Function to get sentence embedding using FastText
def get_sentence_embedding(sentence, model):
    words = sentence.split()
    word_embeddings = [model[word] for word in words if word in model.words]
    
    if not word_embeddings:
        return np.zeros(model.get_dimension())  # Return a zero vector if no valid words found
    
    sentence_embedding = np.mean(word_embeddings, axis=0)
    return sentence_embedding

# Define model name and subset data (first 5000 elements)
subset_text_list = text_list[:5000]
subset_id_list = id_list[:5000]
subset_label_list = label_list[:5000]

# Start timing
start_time = time.time()

# Calculate embeddings for the first 5000 elements
embeddings = [get_sentence_embedding(text, model) for text in subset_text_list]

# End timing
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Time taken to encode 5000 elements: {elapsed_time} seconds")

# Convert embeddings to a DataFrame and add id and label columns
embeddings_df = pd.DataFrame(embeddings)
embeddings_df.insert(0, 'id', subset_id_list)   # Insert 'id' as the first column
embeddings_df['label'] = subset_label_list      # Add 'label' as the last column

# Save the embeddings DataFrame to a CSV file
output_file = "Encode_" + model_name + "_5000_lines.csv"

embeddings_df.to_csv(output_file, index=False)
print(f"Embeddings saved to {output_file}")

# Save timing information to a separate CSV file
timing_df = pd.DataFrame({
    "Model": [model_name],
    "Encoded_Lines": [5000],
    "Time_Seconds": [elapsed_time]
})
timing_file = "Encoding_Time_" + model_name + "_5000_lines.csv"

timing_df.to_csv(timing_file, index=False)
print(f"Encoding time saved to {timing_file}")




'''
# Load the FastText model from Hugging Face Hub
model_name = "fasttext-tr"
model_path = hf_hub_download(repo_id="facebook/fasttext-tr-vectors", filename="model.bin")
model = fasttext.load_model(model_path)

# Function to get sentence embedding using FastText
def get_sentence_embedding(sentence, model):
    words = sentence.split()
    word_embeddings = [model[word] for word in words if word in model.words]
    
    if not word_embeddings:
        return np.zeros(model.get_dimension())  # Return a zero vector if no valid words found
    
    sentence_embedding = np.mean(word_embeddings, axis=0)
    return sentence_embedding

# Calculate embeddings for the text data
embeddings = [get_sentence_embedding(text, model) for text in text_list]

# Convert embeddings to a DataFrame and add id and label columns
embeddings_df = pd.DataFrame(embeddings)
embeddings_df.insert(0, 'id', id_list)  # Insert 'id' as the first column
embeddings_df['label'] = label_list     # Add 'label' as the last column

# Save the complete DataFrame to a CSV file
output_file = "Encode_" + model_name + "_lines.csv"
embeddings_df.to_csv(output_file, index=False)

# Print the shape of the embeddings
print(embeddings_df.shape)  # Shows the DataFrame's shape
print(f"Embeddings saved to {output_file}")
'''