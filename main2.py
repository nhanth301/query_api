from pysimilar import compare
from main import connect_db

def load_titles():
    df = connect_db()
    return [(row['id'],row['title']) for i,row in df.iterrows()]

def queryText(text, titles, n = 5):
    scores = [(i,compare(text,title)) for i,title in titles]
    sorted_score = sorted(scores, key=lambda x: x[1], reverse=True)
    return [i for (i,score) in sorted_score[:n] if score > 0]