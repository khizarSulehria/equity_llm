import pandas as pd
import faiss
pd.set_option('display.max_colwidth', 500)
data = pd.read_csv("sample_text.csv")
print(data.shape)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")


vectors = model.encode(data.text)
print(vectors.shape)


dim = vectors.shape[1]


index = faiss.IndexFlatL2(dim)

index.add(vectors)

print(index.ntotal)

my_query = "eating apple will improve our health"

vec_res = model.encode(my_query)

print(vec_res.shape)

import numpy as np

d_vec = np.array(vec_res).reshape(1,-1)
print(d_vec)

print(index.search(d_vec,k=2))

print(data.loc[[1,0]])