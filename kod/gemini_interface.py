import os
import google.generativeai as genai
from dotenv import load_dotenv
import csv
from collections import defaultdict

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the generation config for the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize the model
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

# Define the prompts
prompt1 = "Yukarıdaki dokümandaki gereksiz kısımları sil ve düzenli bir üniversite ders notu formatında tekrar yaz."
prompt2 = "Yukarıdaki metni, bir gazete makalesi formatında yaz. Yazıyı, haber formatına uygun olarak başlık, alt başlık ve paragraflara ayır. Okuyucuyu bilgilendiren ve dikkatini çeken bir dil kullan, haberin özünü hızlıca açıklayan bir girişle başla ve ardından konuya dair derinlemesine bilgi ver. Makale sonunda konuyla ilgili önemli sonuçlar veya öneriler sun."
prompt3 = "Yukarıdaki dokümanı, bir romanın anlatım tarzında yeniden yazın. Olayları daha akıcı bir şekilde anlatın, duygusal bir ton katın ve karakterlerin bakış açısından anlatmaya çalışın."
prompt4 = "Yukarıdaki metni, bir blog yazısı formatında, geniş bir okuyucu kitlesine hitap edecek şekilde düzenle. Dilini samimi, akıcı ve anlaşılır tut, aynı zamanda konuyu merak uyandırıcı ve ilgi çekici bir şekilde sun. Paragrafları kısa tutarak okunabilirliği artır, başlıklar ve alt başlıklar ekleyerek yazının yapısını belirginleştir. Örnekler ve anekdotlar ile konuyu daha kişisel ve günlük yaşamla ilişkilendirerek okuyucunun dikkatini çek."

# Specify the path to your input CSV file and output files
file_path = 'filtered_doc2line_Filtered_CulturaX_0.csv'
output_file_1 = 'output_prompt1.csv'
output_file_2 = 'output_prompt2.csv'
output_file_3 = 'output_prompt3.csv'
output_file_4 = 'output_prompt4.csv'

# Define a mapping between prompts and their respective output files
prompt_to_output_file = {
    prompt1: output_file_1,
    prompt2: output_file_2,
    prompt3: output_file_3,
    prompt4: output_file_4,
}


max_documents = 2000

csv.field_size_limit(1000000)  # Increase limit to 1MB (adjust as needed)


def read_file_and_process(file_path, max_documents=None, min_document_size=20, skip_documents=11000):
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            
            # Skip header
            next(reader)
            print("File opened successfully, starting to read lines...")

            # Dictionary to group rows by `id1` (document identifier)
            documents = defaultdict(str)  # Store concatenated text for each document

            # Process each row in the CSV
            for row in reader:
                document_id = row[0].split('_')[0]  # Extract `id1` (document identifier)
                input_text = row[1]  # The text column (adjust if necessary)

                # Concatenate the text for rows that have the same `id1`
                documents[document_id] += input_text + "|n"  # Add a space between texts

            print(f"Processed {len(documents)} documents. Grouping by document...")

            # Track the number of valid documents processed
            valid_documents_processed = 0

            # Create a list of documents in a round-robin fashion (distribute them across different prompts)
            prompts = [prompt1, prompt2, prompt3, prompt4]
            document_ids = list(documents.keys())

            # Skip the specified number of documents
            document_ids = document_ids[skip_documents:]
            print(f"Skipping the first {skip_documents} documents. Remaining: {len(document_ids)}")

            for i, document_id in enumerate(document_ids):
                combined_document = documents[document_id]

                # Skip the document if it's too short
                if len(combined_document) < min_document_size:
                    print(f"Skipping document {document_id} because it's too short.")
                    continue  # Skip to the next document

                # Select a prompt in a round-robin fashion
                selected_prompt = prompts[i % len(prompts)]
                # Get the corresponding output file for the selected prompt
                output_file = prompt_to_output_file[selected_prompt]

                # Process the selected document with the assigned prompt
                new_input_text = str(combined_document) + str(selected_prompt)  # Ensure both parts are strings
                print(f"New Input text for {document_id}: {new_input_text}")  # Debugging input text

                # Process the text with the model
                output_text = process_input(new_input_text)
                print(f"Output from model: {output_text}")  # Debugging output from model

                # Save the result in the respective output file
                save_to_file(output_file, document_id, combined_document, output_text, selected_prompt)

                # Increment valid document count after processing
                valid_documents_processed += 1

                # Stop if we have processed enough valid documents
                if max_documents and valid_documents_processed >= max_documents:
                    print(f"Processed {valid_documents_processed} valid documents. Stopping.")
                    break  # Exit the loop once the max number of documents has been processed

            # If the loop ends but max_documents wasn't met, ensure we inform the user
            if max_documents and valid_documents_processed < max_documents:
                print(f"Processed {valid_documents_processed} valid documents. Max documents not reached.")

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")


# Function to process each input and send it to the model
def process_input(new_input_text):
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(new_input_text)
        return response.text
    except Exception as e:
        print(f"Error during model interaction: {e}")
        return "Error generating response"

def save_to_file(output_file, document_id, combined_document, output_text, selected_prompt):
    try:
        # Open the CSV file in append mode
        with open(output_file, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Write the document's ID, combined document, the output text, and the selected prompt
            writer.writerow([document_id, combined_document, output_text])
        
        print(f"Saved output for document {document_id}.")
    except Exception as e:
        print(f"Error saving to file: {e}")



# Read the file and process each row
read_file_and_process(file_path, max_documents)
