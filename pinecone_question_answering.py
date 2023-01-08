import pandas as pd
import openai
import numpy as np
import pickle
from transformers import GPT2TokenizerFast
from transformers import AutoTokenizer
import os
import pinecone
from dotenv import load_dotenv

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "curie"

openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COMPLETIONS_MODEL = "text-davinci-003"

# QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-query-001"
QUERY_EMBEDDINGS_MODEL = "text-embedding-ada-002"

MAX_SECTION_LEN = 3000
SEPARATOR = "\n* "
NO_KNOWLEDGE_STRING = "Sorry, I don't know. I can only construct a response based on transcripts from Huberman Lab episodes and I can't find an answer. If it's helpful, my corpus only includes episodes where Dr. Huberman appears alone. I currently can't parse interview episodes."

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
separator_len = len(tokenizer.tokenize(SEPARATOR))

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}

def request_pinecone_documents(query: str):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    pinecone.init(
        api_key= PINECONE_API_KEY,
        environment="us-west1-gcp"
    )

    xq = openai.Embedding.create(input=query, engine=QUERY_EMBEDDINGS_MODEL)['data'][0]['embedding']
    index = pinecone.Index('transcripts')
    res = index.query([xq], top_k=5, include_metadata=True, namespace='hubermanlab')

    return res["matches"]

def construct_prompt(question: str) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = request_pinecone_documents(question)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
    sources = []

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
     
    for section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        # document_section = df.loc[section_index]
        
        document_section = section_index['metadata']['text']
        title = section_index['metadata']['episode_title']
        position = float(section_index['metadata']['position'])

        chosen_sections_len += len(tokenizer.tokenize(document_section)) + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
        sources.append({"title": title, "position": position, "text": document_section})
            
    # Useful diagnostic information
    # print(f"Selected {len(chosen_sections)} document sections:")
    # print("\n".join(chosen_sections_indexes))
    
    header = f"""Answer the question as truthfully as possible based on the context given below, and if you're unsure of the answer from the context, say "{NO_KNOWLEDGE_STRING}".\n\nContext:\n"""
    
    prompt = header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

    print("Prompt: " + prompt)
    return {"prompt": prompt, "sources": sources}


def answer_question(question: str) -> str:
    res = construct_prompt(question)
    """
    Generate an answer to the supplied question.
    """
    response = openai.Completion.create(
        prompt=res["prompt"],
        **COMPLETIONS_API_PARAMS
    )

    sources = res["sources"]

    if response["choices"][0]["text"].find(NO_KNOWLEDGE_STRING) != -1:
        sources = []
    
    return {"answer": response["choices"][0]["text"], "sources": sources}
