from pysimilar import compare
from main import connect_db, cosine_similarity
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
import pickle

def load_titles():
    df = connect_db()
    return [(row['id'],row['title']) for i,row in df.iterrows()]

def text_embedding(titles):
    model = SentenceTransformer('dangvantuan/vietnamese-embedding')
    embeddings = [(i,model.encode(tokenize(title))) for i,title in titles]
    with open('embeddings.pkl', 'wb') as file:
        pickle.dump(embeddings, file)

def queryText(embeded_text, embeddings, n = 5):
    scores = [(i,cosine_similarity(embeded_text,embed)) for i,embed in embeddings]
    sorted_score = sorted(scores, key=lambda x: x[1], reverse=True)
    return [i for (i,score) in sorted_score[:n] if score > 0]

def load_embedd():
    with open('embeddings.pkl', 'rb') as file:
        embedd = pickle.load(file)
    return embedd