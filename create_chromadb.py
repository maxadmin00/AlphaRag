from chonkie.pipeline import Pipeline
from langchain_chroma import Chroma
import json
from langchain_community.embeddings import OllamaEmbeddings

DOCS_DIR = "./docs"
CHROMA_PATH = "./alpha_chroma_db"
COLLECTION_NAME = "websites_data"

embeddings = OllamaEmbeddings(
    model="jeffh/intfloat-multilingual-e5-small:f32",
)

docs = (Pipeline()
    .fetch_from("file", dir=DOCS_DIR)
    .process_with("text")
    #.chunk_with("sentence",tokenizer="character",  chunk_size=2048,   chunk_overlap=128, min_sentences_per_chunk=1)
    .chunk_with("semantic", embedding_model="minishlab/potion-multilingual-128M", threshold=0.8, chunk_size=2048, similarity_window=3, skip_window=0)
    #.refine_with("embeddings", embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    .refine_with("overlap",tokenizer="character", context_size=0.15, method="prefix", merge=True)
    .run())

print(f"Processed {len(docs)} documents")
for doc in docs:
    print(f"  - {len(doc.chunks)} chunks")

with open('data/websites_dict.json', 'r', encoding='utf-8') as f:
    WEBSITES_DATA = json.load(f)

texts_with_prefix = []
metadatas = []
k = 0
for doc in docs:
    for chunk in doc.chunks:
        chunk_text = "passage: " + chunk.text
        doc_metadata = WEBSITES_DATA[k]['metadata']
        texts_with_prefix.append(chunk_text)
        metadatas.append(doc_metadata)
    k+=1
    #if k > 20:
    #    break

chroma_db = Chroma.from_texts(
    texts=texts_with_prefix,
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory=CHROMA_PATH,
    collection_name=COLLECTION_NAME,
)

