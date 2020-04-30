import pandas as pd
import scipy.spatial
from sentence_transformers import SentenceTransformer

def main(embedding_path, sentence):

    sentence_vectors = pd.read_csv(embedding_path)

    model_path = "content/training_bert_japanese"
    model = SentenceTransformer(model_path, show_progress_bar=True)
    

    query_embedding = model.encode(sentence)

    closest_n = 10
    distances = scipy.spatial.distance.cdist(query_embedding, sentence_vectors, metric="cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for idx, distance in results[0:closest_n]:
        print(sentences[idx].strip(), "(Score: %.4f)" % (distance / 2))

"""
parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')    # 2. パーサを作る

# 3. parser.add_argumentで受け取る引数を追加していく
parser.add_argument('-ep', '--embedding_path', default='content/article_body_embeddings.csv' ,help='この引数の説明（なくてもよい）')    # 必須の引数を追加
parser.add_argument('-s', '--sentenct', default='aaa')

args = parser.parse_args()

main(**args)
"""
