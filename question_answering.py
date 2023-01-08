import pandas as pd
import openai
import numpy as np
import pickle
from transformers import GPT2TokenizerFast
from transformers import AutoTokenizer
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "curie"

COMPLETIONS_MODEL = "text-davinci-003"

# QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-query-001"
QUERY_EMBEDDINGS_MODEL = "text-embedding-ada-002"

MAX_SECTION_LEN = 3000
SEPARATOR = "\n* "

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
separator_len = len(tokenizer.tokenize(SEPARATOR))

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}


def load_embeddings(fname: str):
    """
    Read the document embeddings and their keys from a CSV.
    
    fname is the path to a CSV with exactly these named columns: 
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """
    
    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "text" and c != "episode_title" and c != "author" and c != "position"])
    return {
           (r.episode_title, r.text, r.author, r.position): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }

def get_embedding(text: str, model: str):
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def get_doc_embedding(text: str):
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)

def get_query_embedding(text: str):
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference. 
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_query_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        # document_section = df.loc[section_index]
        
        document_section = section_index[1]
        title = section_index[0]
        position = section_index[3]

        chosen_sections_len += len(tokenizer.tokenize(document_section)) + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.replace("\n", " ") + "Source:" + title + ".Position:" + str(round(position * 100, 0))+"%")
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    # print(f"Selected {len(chosen_sections)} document sections:")
    # print("\n".join(chosen_sections_indexes))
    
    header = """Answer the question as truthfully as possible, and if you're unsure of the answer, say "Sorry, I don't know". End your response with: 'If you'd like to learn more about this, you can find information here:[source] around [position] of the way through the episode'. Then list the main source and position used from the context to generate the answer.\n\nContext:\n"""
    
    prompt = header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

    print("Prompt: " + prompt)
    return prompt


def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")


def main():
    df = pd.read_csv('all_transcripts_embeddings.csv')
    df = df.set_index(["text", "episode_title", "author", "position"]) 
    document_embeddings = load_embeddings('all_transcripts_embeddings.csv')

    while True:
        # Prompt the user for input.
        question = input("What question do you have for Dr. Huberman? ")
        
        # Call the huberman() function to generate a response.
        response = answer_query_with_context(question, df, document_embeddings)
        
        # Print the response to the console.
        print(response)


    # response = answer_query_with_context("What's the 85% rule?", df, document_embeddings)
    # print(response)
    # response = answer_query_with_context("What is learning?", df, document_embeddings)
    # print(response)
    # response = answer_query_with_context("What is delay discounting?", df, document_embeddings)
    # print(response)

    # example_entry = list(document_embeddings.items())[0]
    # print(f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")
    # similiarities = order_document_sections_by_query_similarity("What's the 85% rule?", document_embeddings)[:5]
    # print(similiarities)
    # convert_csv_to_embeddings_csv('nothing')


if __name__ == "__main__":
    main()


# {0: [0.009069414809346199, -0.006452879402786493, -0.012989562936127186, 0.015587475150823593, -0.020373594015836716, -0.009991255588829517, -0.005470514763146639, -0.02124887704849243, -0.003207816742360592, ...]}