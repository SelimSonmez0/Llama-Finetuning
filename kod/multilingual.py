import pandas as pd
from sentence_transformers import SentenceTransformer
import time

# Specify the path to your CSV file
file_path = 'Culturax_linesWithLabels_June_df (3).csv'

# Read the CSV file, allowing for variable number of columns and no header
df = pd.read_csv(file_path, header=None, na_filter=False)

# Remove any entirely empty columns
df = df.dropna(axis=1, how='all')

# Prepare lists for id, text, label, and the 8th element from the end
id_list = []
text_list = []
label_list = []


# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    
    if index == 0:  # Skip the first row
        continue

    id_value = row[0]            # First column for ID
   
    # Initialize label_value to None
    label_value = None
    
    # Start from the last element and move backward to find the first non-empty label
    for i in range(len(row) - 1, -1, -1):
        if row[i] != '':
            label_value = row[i]
            break
        
        
     # Check if label is binary
    if label_value not in ['0', '1']:
        
        print(f"Label={label_value} for ID {id_value} is not binary. Skipping this row.")  # Notify about skipping
        input()
        continue  # Skip this row if the label is not binary
        
        
        
    # Gather all text elements between ID and label
    text_value = ','.join(row[1:-12].astype(str))  # Convert all text elements to string and join them
    
    # Append values to their respective lists
    id_list.append(id_value)
    text_list.append(text_value)
    label_list.append(label_value)

# Print a sample of the data
print("Sample data:")
for i in range(min(110, len(id_list))):  # Ensure we do not exceed list length
    print(f"ID: {id_list[i]}, Text: {text_list[i]}, Label: {label_list[i]}")
    






# Load the multilingual sentence transformer model
model_name = "distiluse-base-multilingual-cased-v1"
model = SentenceTransformer(model_name)


line_count=25000

# Slice ID and label lists to match the line_count
subset_id_list = id_list[:line_count]
subset_label_list = label_list[:line_count]

# Assuming `text_list` contains your text data and `model` is already defined
subset_text_list = text_list[:line_count]  # Get the first 5000 elements


# Start the timer
start_time = time.time()

# Calculate embeddings for the first 5000 elements
embeddings = model.encode(subset_text_list)

# End the timer
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Time taken to encode {line_count} elements: {elapsed_time} seconds")


# Convert embeddings to a DataFrame
embeddings_df = pd.DataFrame(embeddings)
embeddings_df.insert(0, 'id', subset_id_list)  # Insert sliced IDs as the first column
embeddings_df['label'] = subset_label_list     # Add sliced labels as the last column



# Save the time to a CSV file
output_file = "Encode_" + model_name + "_lines.csv"





embeddings_df.to_csv(output_file, index=False)



# Create a DataFrame to store the results
timing_df = pd.DataFrame({
    "Model": [model_name],
    "Encoded_Lines": [line_count],
    "Time_Seconds": [elapsed_time]
})


timing_file = f"Encoding_Time_{model_name.replace('/', '_')}_{line_count}_lines.csv"



# Save the DataFrame to a CSV file
timing_df.to_csv(timing_file, index=False)
print(f"Encoding time saved to {output_file}")



'''


# Load the multilingual sentence transformer model
model_name = "distiluse-base-multilingual-cased-v1"
model = SentenceTransformer(model_name)


# Calculate embeddings for the text data
embeddings = model.encode(text_list)

# Convert embeddings to a DataFrame and add id and label columns
embeddings_df = pd.DataFrame(embeddings)
embeddings_df.insert(0, 'id', id_list)  # Insert 'id' as the first column
embeddings_df['label'] = label_list     # Add 'label' as the last column

# Save the complete DataFrame to a CSV file
output_file = "Encode_" + model_name + "_lines.csv"
embeddings_df.to_csv(output_file, index=False)

# Print the shape of the embeddings
print(embeddings.shape)  # Should show the shape, e.g., [number_of_sentences, embedding_size]
print(f"Embeddings saved to {output_file}")

'''