import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

def load_model():
    model_start_loading = time.time()
    print("* Model Loading 。。。")
    model = SentenceTransformer("maidalun1020/bce-embedding-base_v1") 
    print(f"* Model Loaded。 |{str(time.time() - model_start_loading)[:5]}s")
    return model

def _txt_strip(text):
    # sentences = [s.strip() for s in text.split('\n') if s.strip()]
    sentences = text_splitter.split_text(text)
    return sentences

def _load_txt():
    with open("./data/content.txt", 'r', encoding='utf-8') as f:
        text = f.read()
    sentences = _txt_strip(text)
    return sentences

def embed_content(model):
    sentences = _load_txt()
    print("* Converting \"content.txt\" to embeddings")
    embeddings = model.encode(sentences)
    np.save('./data/embeddings.npy', embeddings)

def _embeddings_to_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings)) # type: ignore
    return index

def _search(query, model, index: faiss.IndexFlatL2, k, textForm_data):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k) # type: ignore
    # print("Query:", query)
    # print("\nTop Matches:")
    results = []
    for i, idx in enumerate(indices[0]):
        if distances[0][i]< 20:        
            results.append(textForm_data[idx])
    return results


def get_rag_embedding(query, model):
    embeddings = np.load('./data/embeddings.npy')
    index = _embeddings_to_index(embeddings)
    results = _search(query, model, index, 5, _load_txt())
    return results
        

if __name__ == '__main__':
    model = load_model()
    # embed_content(model)
    embeddings = np.load('./data/embeddings.npy')
    index = _embeddings_to_index(embeddings)
    results = _search("我的名字是什麽？", model, index, 5, _load_txt())
    if len(results) > 0:
        print(results[0])