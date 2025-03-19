import pandas as pd
import string
import re

import nltk
from nltk.corpus import words


# Specify the path to your CSV file
file_path = 'Culturax_linesWithLabels_June_df (3).csv'

file_path = 'culturax_documents_with_labels.csv'



# Read the CSV file, skipping the first row
df = pd.read_csv(file_path, header=None, na_filter=False, skiprows=1)

# Remove any entirely empty columns
df = df.dropna(axis=1, how='all')

# Prepare lists for id, text, label, and the 8th element from the end
id_list = []
text_list = []
label_list = []


# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    

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
    
print(len(df))  # Length of the DataFrame
print(len(text_list))  # Length of the text_list

print(df.head())  # Look at the first few rows of the DataFrame
print(text_list[:5])  # Print the first 5 elements of text_list


df['id'] = id_list
df['text'] = text_list
df['label'] = label_list





# Count the number of words in the 'text' column for each row
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

# Count the number of characters in the 'text' column
df['character_count'] = df['text'].apply(lambda x: len(str(x)))

# Count the number of sentences in the 'text' column
df['sentence_count'] = df['text'].apply(lambda x: len(re.split(r'[.!?]+', str(x).strip())) - 1)

# Find the length of the longest word in the 'text' column
df['longest_word_length'] = df['text'].apply(lambda x: max((len(word) for word in str(x).split()), default=0))

# Find the length of the shortest word in the 'text' column
df['shortest_word_length'] = df['text'].apply(lambda x: min((len(word) for word in str(x).split()), default=0))

# Calculate the average sentence length (number of words per sentence)
df['average_sentence_length'] = df.apply(lambda row: row['word_count'] / row['sentence_count'] 
                                         if row['sentence_count'] > 0 else 0, axis=1)

df['average_word_length'] = df['character_count'] / df['word_count']


# Calculate the ratio of uppercase letters to character count (excluding spaces)
df['upper_case_ratio'] = df['text'].apply(
    lambda x: sum(1 for char in str(x) if char.isupper()) / len(str(x).replace(" ", ""))
    if len(str(x).replace(" ", "")) > 0 else 0
)

# Calculate the ratio of lowercase letters to character count (excluding spaces)
df['lower_case_ratio'] = df['text'].apply(
    lambda x: sum(1 for char in str(x) if char.islower()) / len(str(x).replace(" ", ""))
    if len(str(x).replace(" ", "")) > 0 else 0
)

# Calculate the ratio of punctuation marks to character count (excluding spaces)
df['punctuation_ratio'] = df['text'].apply(
    lambda x: sum(1 for char in str(x) if char in string.punctuation) / len(str(x).replace(" ", ""))
    if len(str(x).replace(" ", "")) > 0 else 0
)

# Calculate the ratio of digits to character count
df['digit_ratio'] = df['text'].apply(
    lambda x: sum(1 for char in str(x) if char.isdigit()) / len(str(x).replace(" ", "")) 
    if len(str(x).replace(" ", "")) > 0 else 0
)

# Calculate the ratio of whitespace characters to character count
df['whitespace_ratio'] = df['text'].apply(
    lambda x: sum(1 for char in str(x) if char.isspace()) / len(str(x).replace(" ", "")) 
    if len(str(x).replace(" ", "")) > 0 else 0
)
# Calculate the ratio of vowels to character count
df['vowel_ratio'] = df['text'].apply(
    lambda x: sum(1 for char in str(x).lower() if char in 'aeiou') / len(str(x).replace(" ", "")) 
    if len(str(x).replace(" ", "")) > 0 else 0
)

# Calculate the ratio of consonants to character count
df['consonant_ratio'] = df['text'].apply(
    lambda x: sum(1 for char in str(x).lower() if char.isalpha() and char not in 'aeiou') / len(str(x).replace(" ", "")) 
    if len(str(x).replace(" ", "")) > 0 else 0
)
# Calculate the ratio of uppercase to lowercase characters
df['upper_lower_ratio'] = df.apply(lambda row: row['upper_case_ratio'] / row['lower_case_ratio'] 
                                    if row['lower_case_ratio'] > 0 else 0, axis=1)


# Calculate the average word length
df['average_word_length'] = df['text'].apply(lambda x: sum(len(word) for word in str(x).split()) / len(str(x).split()) 
                                             if len(str(x).split()) > 0 else 0)

# Calculate lexical diversity (ratio of unique words to total word count)
df['unique_word_ratio'] = df['text'].apply(lambda x: len(set(str(x).split())) / len(str(x).split()) 
                                           if len(str(x).split()) > 0 else 0)

# Calculate the ratio of words in all caps (shouting) to the total word count
df['all_caps_ratio'] = df['text'].apply(lambda x: sum(1 for word in str(x).split() if word.isupper()) / len(re.findall(r'\b\w+\b', str(x))) if len(re.findall(r'\b\w+\b', str(x))) > 0 else 0)

# Calculate ratios for words ending with 'm', 'z', second-to-last character 'd', and ending with 'ş'
df['words_ending_with_m_ratio'] = df['text'].apply(
    lambda x: sum(1 for word in str(x).split() if word.endswith('m')) / len(re.findall(r'\b\w+\b', str(x)))
    if len(re.findall(r'\b\w+\b', str(x))) > 0 else 0
)

df['words_ending_with_z_ratio'] = df['text'].apply(
    lambda x: sum(1 for word in str(x).split() if word.endswith('z')) / len(re.findall(r'\b\w+\b', str(x)))
    if len(re.findall(r'\b\w+\b', str(x))) > 0 else 0
)

df['words_second_last_char_d_ratio'] = df['text'].apply(
    lambda x: sum(1 for word in str(x).split() if len(word) > 1 and word[-2] == 'd') / len(re.findall(r'\b\w+\b', str(x)))
    if len(re.findall(r'\b\w+\b', str(x))) > 0 else 0
)

df['words_ending_with_sh_ratio'] = df['text'].apply(
    lambda x: sum(1 for word in str(x).split() if word.endswith('ş')) / len(re.findall(r'\b\w+\b', str(x)))
    if len(re.findall(r'\b\w+\b', str(x))) > 0 else 0
)

# Calculate the ratio of four-digit sequences to total word count without adding an intermediary column
df['four_digit_sequences_ratio'] = df['text'].apply(
    lambda x: len(re.findall(r'\b\d{4}\b', str(x))) / len(str(x).split()) if len(str(x).split()) > 0 else 0
)

# Calculate 'words_ending_with_r_ratio' as the ratio of words ending with 'r' to the total word count in the 'text' column
df['words_ending_with_r_ratio'] = df['text'].apply(
    lambda x: len(re.findall(r'\b\w*r\b', str(x))) / len(re.findall(r'\b\w+\b', str(x))) 
    if len(re.findall(r'\b\w+\b', str(x))) > 0 else 0
)

df['de_da_ratio'] = df['text'].apply(
    lambda x: len(re.findall(r'\b(de|da)\b', str(x), re.IGNORECASE)) / len(re.findall(r'\b\w+\b', str(x))) 
    if len(re.findall(r'\b\w+\b', str(x))) > 0 else 0
)

# Calculate 'yok_ratio' as the ratio of 'yok' occurrences to the total word count in the 'text' column
df['yok_ratio'] = df['text'].apply(
    lambda x: len(re.findall(r'\byok\b', str(x), re.IGNORECASE)) / len(re.findall(r'\b\w+\b', str(x))) 
    if len(re.findall(r'\b\w+\b', str(x))) > 0 else 0
)


# Calculate 'MÖ_ratio' as the ratio of 'M.Ö.' occurrences to the total word count in the 'text' column
df['MÖ_ratio'] = df['text'].apply(
    lambda x: len(re.findall(r'\bM\.Ö\.\b', str(x))) / len(re.findall(r'\b\w+\b', str(x))) if len(re.findall(r'\b\w+\b', str(x))) > 0 else 0
)

# Calculate 'emoji_ratio' as the ratio of emoji occurrences to the total character count in the 'text' column
df['emoji_ratio'] = df['text'].apply(lambda x: len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FAFF\U00002700-\U000027BF]', str(x))) / len(str(x)) if len(str(x)) > 0 else 0)

# Calculate 've_ratio' as the ratio of 've' occurrences to total word count in the 'text' column
df['en_ratio'] = df['text'].apply(lambda x: len(re.findall(r'\ben\b', str(x), re.IGNORECASE)) / len(str(x).split()) if len(str(x).split()) > 0 else 0)

# Calculate 've_ratio' as the ratio of 've' occurrences to total word count in the 'text' column
df['ve_ratio'] = df['text'].apply(lambda x: len(re.findall(r'\bve\b', str(x), re.IGNORECASE)) / len(str(x).split()) if len(str(x).split()) > 0 else 0)

# Add 'com_ratio' directly by dividing the count of 'com' by the total character count in each row
df['com_ratio'] = df['text'].apply(lambda x: x.count('com') / len(x) if len(x) > 0 else 0)

# Create the ratio column directly without needing the 'english_not_turkish_count' column
english_not_turkish_chars = set('wqx')
df['english_not_turkish_ratio'] = df['text'].apply(
    lambda x: sum(1 for char in str(x) if char in english_not_turkish_chars) / len(str(x))
    if len(str(x)) > 0 else 0
)



# Define the pronouns for substring counting
pronouns = ["ben", "sen", "biz", "o", "siz", "onlar"]

# Function to calculate the substring ratio for each pronoun
def calculate_ratio(pronoun, text):
    word_count = len(str(text).split())
    if word_count == 0:
        return 0
    return len(re.findall(rf'{pronoun}', str(text), re.IGNORECASE)) / word_count

# Add a column for each pronoun ratio with substring counting
for pronoun in pronouns:
    df[f'{pronoun}_ratio'] = df['text'].apply(lambda x: calculate_ratio(pronoun, x))






# List of common Turkish time references
time_references_tr = set([
    "dün", "bugün", "yarın", "sabah", "öğle", "akşam", "gece",
    "Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar",
    "hafta", "ay", "yıl", "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
    "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık", "saat", "dakika", "saniye"
])

# Function to calculate the time reference ratio
def calculate_time_reference_ratio(text):
    words = text.split()
    time_reference_count = sum(1 for word in words if word in time_references_tr)
    total_words = len(words)
    return time_reference_count / total_words if total_words > 0 else 0


# Apply the function to each row in the DataFrame
df['time_reference_ratio'] = df['text'].apply(calculate_time_reference_ratio)


# List of common Turkish stop words categorized into positive, negative, and neutral
positive_words = set(["çok", "yine"])  # Positive words
negative_words = set(["ama", "ancak"])  # Negative words
neutral_words = set([
    "ve", "bir", "bu", "ne", "nasıl", "için", "da", "de", "ile", 
    "ki", "gibi", "ya", "şu", "üzerine", "hala", "bunu", "önce", "sonra", "gibi", "şey"
])  # Neutral words

# Function to calculate the stop words ratio for each category (positive, negative, neutral)
def calculate_category_stop_words_ratio(text):
    # Ensure that the text is not None or empty
    if not text:
        return 0, 0, 0
    
    words = text.split()
    
    # Count occurrences of each category of words
    positive_count = sum(1 for word in words if word.lower() in positive_words)
    negative_count = sum(1 for word in words if word.lower() in negative_words)
    neutral_count = sum(1 for word in words if word.lower() in neutral_words)
    
    # Total word count
    total_words = len(words)
    
    # Calculate ratios for each category
    positive_ratio = positive_count / total_words if total_words > 0 else 0
    negative_ratio = negative_count / total_words if total_words > 0 else 0
    neutral_ratio = neutral_count / total_words if total_words > 0 else 0
    
    return positive_ratio, negative_ratio, neutral_ratio

# Apply the function to each row in the DataFrame
df[['positive_stop_words_ratio', 'negative_stop_words_ratio', 'neutral_stop_words_ratio']] = df['text'].apply(
    lambda x: pd.Series(calculate_category_stop_words_ratio(x))
)




# Make sure you have the words corpus downloaded
nltk.download('words')

# List of English words from nltk corpus
english_words = set(words.words())

# Function to calculate the ratio of English words in the text
def calculate_english_words_ratio(text):
    if not text:
        return 0
    
    words_in_text = text.split()
    
    # Count words that are in the English words list (ignoring case)
    english_count = sum(1 for word in words_in_text if word.lower() in english_words)
    
    # Total word count
    total_words = len(words_in_text)
    
    # Calculate the ratio of English words to total words
    english_ratio = english_count / total_words if total_words > 0 else 0
    
    return english_ratio

# Apply the function to each row in the DataFrame
df['english_word_ratio'] = df['text'].apply(calculate_english_words_ratio)





# Calculate the ratio of occurrences of substrings 'me' or 'ma' to the total character count (excluding spaces)
df['me_ratio'] = df['text'].apply(
    lambda x: len(re.findall(r'me', str(x), re.IGNORECASE)) / len(str(x).replace(" ", "")) 
    if len(str(x).replace(" ", "")) > 0 else 0
)

# Calculate the ratio of occurrences of substrings 'me' or 'ma' to the total character count (excluding spaces)
df['ma_ratio'] = df['text'].apply(
    lambda x: len(re.findall(r'ma', str(x), re.IGNORECASE)) / len(str(x).replace(" ", "")) 
    if len(str(x).replace(" ", "")) > 0 else 0
)



def compute_ratio(text):
    # Convert the text to lowercase
    text = str(text).lower()
    
    # Find suffixes "me" and "ma" (not followed by a letter)
    me_count = len(re.findall(r'me(?![a-zA-Z])', text))  # 'me' as a suffix
    ma_count = len(re.findall(r'ma(?![a-zA-Z])', text))  # 'ma' as a suffix
    
    # Split the text into words and count the total number of words
    words = re.findall(r'\w+', text)  # This captures words
    total_words = len(words)
    
    # Calculate ratios (if total words is 0 to avoid division by zero)
    me_suffix_ratio = me_count / total_words if total_words > 0 else 0
    ma_suffix_ratio = ma_count / total_words if total_words > 0 else 0
    
    return me_suffix_ratio, ma_suffix_ratio

# Apply this function to each row of text
df[['me_suffix_ratio', 'ma_suffix_ratio']] = df['text'].apply(lambda x: pd.Series(compute_ratio(x)))




# Display the DataFrame with the updated pronoun ratio columns
print(df[['id', 'text', 'word_count', 'character_count', 'sentence_count',
           'longest_word_length', 'shortest_word_length', 'average_sentence_length',
           'average_word_length', 'upper_case_ratio', 'lower_case_ratio', 'punctuation_ratio', 
           'digit_ratio', 'whitespace_ratio', 'vowel_ratio', 'consonant_ratio', 
           'upper_lower_ratio', 'unique_word_ratio', 'all_caps_ratio', 'words_ending_with_m_ratio', 
           'words_ending_with_z_ratio', 'words_second_last_char_d_ratio', 'words_ending_with_sh_ratio', 
           'four_digit_sequences_ratio', 'words_ending_with_r_ratio', 'de_da_ratio', 'yok_ratio', 
           'MÖ_ratio', 'emoji_ratio', 'en_ratio', 've_ratio', 'com_ratio', 'english_not_turkish_ratio',
           'ben_ratio', 'sen_ratio', 'biz_ratio', 'o_ratio', 'siz_ratio', 'onlar_ratio', 'time_reference_ratio',
           'positive_stop_words_ratio', 'negative_stop_words_ratio', 'neutral_stop_words_ratio','english_word_ratio',
           'me_ratio','ma_ratio','me_suffix_ratio','ma_suffix_ratio']])



# List of the specific columns you want to include
columns_to_save = [
    'id', 'word_count', 'character_count', 'sentence_count', 'longest_word_length', 'shortest_word_length',
    'average_sentence_length', 'average_word_length', 'upper_case_ratio', 'lower_case_ratio', 'punctuation_ratio', 
    'digit_ratio', 'whitespace_ratio', 'vowel_ratio', 'consonant_ratio', 'upper_lower_ratio', 'unique_word_ratio',
    'all_caps_ratio', 'words_ending_with_m_ratio', 'words_ending_with_z_ratio', 'words_second_last_char_d_ratio', 
    'words_ending_with_sh_ratio', 'four_digit_sequences_ratio', 'words_ending_with_r_ratio', 'de_da_ratio', 'yok_ratio', 
    'MÖ_ratio', 'emoji_ratio', 'en_ratio', 've_ratio', 'com_ratio', 'english_not_turkish_ratio', 'ben_ratio', 'sen_ratio', 
    'biz_ratio', 'o_ratio', 'siz_ratio', 'onlar_ratio', 'time_reference_ratio', 'positive_stop_words_ratio', 
    'negative_stop_words_ratio', 'neutral_stop_words_ratio', 'english_word_ratio', 'me_ratio','ma_ratio','me_suffix_ratio','ma_suffix_ratio',
    'label'
]

# Ensure your DataFrame only includes the desired columns
df_to_save = df[columns_to_save]

# Specify the output file path
output_file_path = 'manual_features.csv'

# Specify the output file path
output_file_path = 'doc_manual_features.csv'


# Save the DataFrame to a CSV file
df_to_save.to_csv(output_file_path, index=False)





