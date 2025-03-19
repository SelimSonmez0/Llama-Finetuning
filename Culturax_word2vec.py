import time
import gc
import pandas as pd
from tqdm import tqdm  # Import tqdm for the progress bar
from gensim.models import KeyedVectors
import numpy as np

# Load word vectors
word_vectors = KeyedVectors.load_word2vec_format('trmodel', binary=True)

# Paths to input and output files
input_file = "doc2line_Filtered_CulturaX_0.csv"
output_file = "embeddings_all.csv"

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

# Process the file in chunks using pandas
def process_in_chunks(chunk_size=500000, batch_size=100000):
    total_rows = sum(1 for _ in open(input_file, encoding="utf-8"))  # Total rows in the CSV
    total_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size != 0 else 0)  # Total chunks

    # Read the file in chunks using pandas
    for chunk_idx in tqdm(range(total_chunks), desc="Processing chunks"):
        # Read a chunk of data into a DataFrame
        df_chunk = pd.read_csv(input_file, skiprows=chunk_idx * chunk_size, nrows=chunk_size, encoding="utf-8")

        # Initialize a list to hold batch results
        batch_embeddings = []
        
        # Process the chunk in batches
        for start in range(0, len(df_chunk), batch_size):
            end = min(start + batch_size, len(df_chunk))
            batch = df_chunk.iloc[start:end]

            # Ensure batch_embeddings is always initialized
            batch_embeddings = []  # Initialize the list for every new batch
            
            # Compute embeddings for the batch
            embedding_start_time = time.time()
            for _, row in batch.iterrows():
                embedding = get_sentence_embedding(row['Text'], word_vectors)
                embedding_dict = {
                    'id': row['New_ID'],
                    **{f'embedding_{i}': value for i, value in enumerate(embedding)}
                }
                batch_embeddings.append(embedding_dict)
            embedding_end_time = time.time()

            # Write the embeddings to the output file
            write_start_time = time.time()
            # Append the batch embeddings to the output CSV
            batch_df = pd.DataFrame(batch_embeddings)
            batch_df.to_csv(output_file, mode='w', header=(chunk_idx == 0 and start == 0), index=False)
            write_end_time = time.time()

            # Log the timing details
            print(f"\nBatch rows {start} to {end}:")
            print(f"- Embedding time: {embedding_end_time - embedding_start_time:.4f} seconds")
            print(f"- Writing time: {write_end_time - write_start_time:.4f} seconds")

            # Clear memory for the batch
            del batch_embeddings
            gc.collect()

    return

# Start timing
start_time = time.time()

# Process the input file in chunks
process_in_chunks(chunk_size=500000, batch_size=100000)

# End timing
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken to encode: {elapsed_time:.2f} seconds")
