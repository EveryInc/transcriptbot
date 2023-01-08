import pandas as pd
import os
import openai
import csv
import pinecone
import string
import random
from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
MODEL_NAME = "ada"

# DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"
DOC_EMBEDDINGS_MODEL = "text-embedding-ada-002"

def generate_unique_id():
    # Generate a random 16-character string using the characters A-Z, a-z, and 0-9
    unique_id = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
    return unique_id

def merge_embeddings_csvs(csv1, csv2, output):
    # Open both input files
    with open(csv1, 'r') as file1, open(csv2, 'r') as file2:
        # Create a reader for each file
        reader1 = csv.reader(file1)
        reader2 = csv.reader(file2)

        # Open the output file
        with open(output, 'w') as output:
            # Create a writer for the output file
            writer = csv.writer(output)

            # Iterate over the rows in both files, and write them to the output file
            for row1, row2 in zip(reader1, reader2):
                writer.writerow(row1 + row2)


def get_number_of_lines_in_file(file_path):
    with open(file_path, 'r') as fp:
        for count, line in enumerate(fp):
            pass
    print('Total Lines', count + 1) 
    return count + 1

def get_number_of_words_in_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()

    # Split the text into a list of words
    words = text.split()
    return len(words)


def read_file_into_text_chunks(file_path, episode_title, author):
    csv_output = {"text": [], "episode_title": [], "author": [], "position": []}
    # Initialize a variable to keep track of the current chunk
    current_chunk = ""

    # Initialize a variable to keep track of the word count
    word_count = 0

    # Check if the file exists
    if os.path.exists(file_path):
        number_of_words = get_number_of_words_in_file(file_path)
        # Open the file
        with open(file_path, "r") as f:
            total_word_count = 0
            # Iterate through each line in the file
            for line in f:
                # Split the line into words
                words = line.split()
                
                # Iterate through each word in the line
                for word in words:
                    # Increment the word count
                    word_count += word.__len__()
                    total_word_count += 1
                    
                    # Add the word to the current chunk
                    current_chunk += word + " "
                    position = float(total_word_count) / float(number_of_words)
                    
                    # If the word count is 1,000, add the current chunk to the array and reset the current chunk and the word count
                    if word_count >= 400:
                        csv_output['text'].append(current_chunk)
                        csv_output['episode_title'].append(episode_title)
                        csv_output['author'].append(author)
                        csv_output['position'].append(position)
                        current_chunk = ""
                        word_count = 0
            
            # If there are any remaining words in the current chunk after the loop is done, add them to the array
            if current_chunk:
                csv_output['text'].append(current_chunk)
                csv_output['episode_title'].append(episode_title)
                csv_output['author'].append(author)
                csv_output['position'].append(position)
            print(csv_output)
    else:
        # If the file does not exist, print an error message
        print("The file could not be found.")
    return(csv_output)

def get_embedding(text: str, model: str):
    print("Getting embedding for:" + text)
    try: 
        result = openai.Embedding.create(
            model=model,
            input=text
        )
        return result["data"][0]["embedding"]
    except Exception as e:
        print ("Error: " + str(e))
        return ""

def get_doc_embedding(text: str):
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)

def get_query_embedding(text: str):
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)

def compute_doc_embeddings(df: pd.DataFrame):
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_doc_embedding(r.text.replace("\n", " ")) for idx, r in df.iterrows()
    }

def convert_csv_to_embeddings_csv(fname: str):
    df = pd.read_csv(fname)
    embeddings = compute_doc_embeddings(df)
    embeddings_df = pd.DataFrame.from_dict(embeddings, orient='index')
    result = pd.concat([df, embeddings_df], axis=1)
    result.to_csv(fname + "_embeddings.csv", index=False)

def get_embedding(text: str, model: str):
    print("Getting embedding for:" + text)
    try: 
        result = openai.Embedding.create(model=model, input=text)
        return result["data"][0]["embedding"]
    except Exception as e:
        print("Error: " + str(e))
        return ""

def read_directory_into_pinecone_embeddings(directory_path):
    # Initialize Pinecone with your API key and environment
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment="us-west1-gcp"
    )
    id = 0

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename)) as file:
                print("Reading file: " + filename + "")
                title = filename.strip(".txt").strip(".mp3").replace("_", " ")
                print("Title: " + title)
                csv_output = read_file_into_text_chunks(os.path.join(directory_path, filename), title, "Andrew Huberman")
                
                # Iterate through the text chunks
                for i in range(len(csv_output['text'])):
                    # Get the text chunk, position, author, and episode title for the current iteration
                    text = csv_output['text'][i]
                    position = csv_output['position'][i]
                    author = csv_output['author'][i]
                    episode_title = csv_output['episode_title'][i]
                    id += 1

                    # Calculate the embedding for the text chunk using OpenAI
                    embedding = get_embedding(text, DOC_EMBEDDINGS_MODEL)

                     # Check if the 'hubermanlab' index already exists (create it if not)
                    if 'transcripts' not in pinecone.list_indexes():
                        pinecone.create_index('transcripts', dimension=len(embedding))

                    # Connect to the 'hubermanlab' index
                    index = pinecone.Index('transcripts')

                    # Format the metadata in the desired format
                    meta = {'text': text, 'position': position, 'author': author, 'show': 'Huberman Lab Podcast', 'episode_title': episode_title}

                    # Save the embedding and meta data to the 'hubermanlab' index in Pinecone
                    index.upsert([(id.__str__(), embedding, meta)], namespace='hubermanlab')



def read_directory_into_text_chunks(directory_path):
    csv_output = {"text": [], "episode_title": [], "author": [], "position": []}
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename)) as file:
                print("Reading file: " + filename + "")
                title = filename.strip(".txt").strip(".mp3").replace("_", " ")
                print("Title: " + title)
                new_csv_output = read_file_into_text_chunks(os.path.join(directory_path, filename), title, "Andrew Huberman")
                
                csv_output['text'].extend(new_csv_output['text'])
                csv_output['episode_title'].extend(new_csv_output['episode_title'])
                csv_output['author'].extend(new_csv_output['author'])
                csv_output['position'].extend(new_csv_output['position'])
                
                # csv_output.update(new_csv_output)
    return csv_output


def main():
    # Read the file into chunks
    # output = read_directory_into_text_chunks("data_preparation/transcripts")

    # # # Create a DataFrame from the chunks
    # df = pd.DataFrame(output, columns=["text", "episode_title", "author", "position"])

    # # # Save the DataFrame to a CSV file
    # df.to_csv("all_transcripts.csv", index=False)

    # print("Converting CSV to embeddings CSV");
    # convert_csv_to_embeddings_csv("all_transcripts.csv")
    # # merge_embeddings_csvs('data_preparation/goals_embeddings.csv', 'data_preparation/habits_embeddings.csv', 'data_preparation/merged_embeddings.csv')

    read_directory_into_pinecone_embeddings("data_preparation/transcripts")
    
if __name__ == "__main__":
    main()