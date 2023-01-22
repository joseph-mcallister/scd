import json
import pandas as pd
from numpy import dot, array
from numpy.linalg import norm

from get_embeddings import get_embedding

def cosine_similarity(a,b):
    assert a.shape == b.shape
    return dot(a,b)/(norm(a)*norm(b))

def search_items(df, item, n=5):
    embedding = array(get_embedding(item))
    df['similarities'] = df.ada_embedding.apply(
        lambda x: cosine_similarity(embedding, array(json.loads(x)))
    )
    res = df.sort_values('similarities', ascending=False).head(n)
    return res
 
if __name__ == "main":
    df = pd.read_csv('legal_list_with_embeddings.csv', sep=":")
    res = search_items(df, 'texas toast')
    print(res)