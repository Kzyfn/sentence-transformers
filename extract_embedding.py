import sys
sys.path.append('/usr/local/lib/python3.7/site-packages/')

from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import argparse


def main(args):

    #prl = pd.read_csv(args.prl_path, names= list(range(58)))#9: title, 10:body
    articles = pd.read_csv(args.article_path)

    model_path = "content/training_bert_japanese"
    model = SentenceTransformer(model_path, show_progress_bar=True)

    #prl_title_embeddings = model.encode(prl[9][1:])
    #article_title_embeddings = model.encode(articles['title'])
    article_body_embeddings = model.encode(articles['body'])

    #pd.DataFrame(prl_title_embeddings).to_csv(args.export_prl_title_path, index=None)
    pd.DataFrame(article_body_embeddings).to_csv(args.export_artile_body_path, index=None)
    
    #del prl_title_embeddings
    #del article_title_embeddings


parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')    # 2. パーサを作る

# 3. parser.add_argumentで受け取る引数を追加していく
parser.add_argument('-eptp', '--export_prl_title_path', default='content/prl_title_embeddings.csv' ,help='この引数の説明（なくてもよい）')    # 必須の引数を追加
parser.add_argument('-eatp', '--export_article_title_path', default='content/article_title_embeddings.csv' ,help='この引数の説明（なくてもよい）')
parser.add_argument('-eabp', '--export_artile_body_path', default='content/article_body_embeddings.csv' ,help='この引数の説明（なくてもよい）')
parser.add_argument('-pl', '--prl_path', default='data/PRL2011_2017.csv' ,help='この引数の説明（なくてもよい）')
parser.add_argument('-al', '--article_path', default='data/articles.csv' ,help='この引数の説明（なくてもよい）')

args = parser.parse_args()

main(args)
