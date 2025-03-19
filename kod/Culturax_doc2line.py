import pandas as pd

# File path of the original file
file_path = 'Filtered_CulturaX_0.csv'

# Read the file with only a Text column
df = pd.read_csv(file_path, names=["Text"], sep=",", quotechar='"', encoding="utf-8")

# Drop rows where the Text is missing (NaN)
df = df.dropna(subset=["Text"])

# Prepare the new data for the output file
new_id_list = []
new_text_list = []

# Iterate through the rows of the DataFrame
for index, row in df.iterrows():
    text = row['Text']  # Full text

    # Split the text into individual lines
    text_lines = text.splitlines()

    # Assign a new ID for each line
    for line_index, text_line in enumerate(text_lines):
        new_id = f"{index + 1}_{line_index + 1}"  # Create a unique ID (e.g., 1_1, 1_2, etc.)
        new_id_list.append(new_id)
        new_text_list.append(text_line)
print(len(new_id_list))
# Create a new DataFrame with the updated structure
new_df = pd.DataFrame({
    "New_ID": new_id_list,
    "Text": new_text_list,
})

# Save the new DataFrame to a CSV file
output_file = f"doc2line_{file_path}"
new_df.to_csv(output_file, index=False, encoding="utf-8")

print(f"New file created: {output_file}")
