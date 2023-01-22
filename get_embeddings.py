import os
import pandas as pd
import openai

openai.api_key = os.environ.get("OPENAI_API_KEY")

def get_embedding(text):
    response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return response["data"][0]["embedding"]

if __name__ == "__main__":
    df = pd.read_csv('legal_list.csv', sep=":")
    with_embeddings = df.copy()
    df['ada_embedding'] = df["item"].apply(lambda x: get_embedding(x))
    df.to_csv("legal_list_with_embeddings.csv", sep=":", index=False)
    