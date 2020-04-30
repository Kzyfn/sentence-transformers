import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece
import pandas as pd

import tensorflow.compat.v1 as tf

def main(args):

    #prl = pd.read_csv(args.prl_path, names= list(range(58)))#9: title, 10:body
    articles = pd.read_csv(args.article_path)
    
    
    g = tf.Graph()
    with g.as_default():
        text_input = tf.placeholder(dtype=tf.string, shape=[None])
        xling_8_embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-xling-many/1")
        embedded_text = xling_8_embed(text_input)
        init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    g.finalize()
    
    # Initialize session.
    session = tf.Session(graph=g)
    session.run(init_op)

    print("load successful")
    article_body_embeddings = session.run(embedded_text, feed_dict={text_input: articles['body']})
    article_title_embeddings = session.run(embedded_text, feed_dict={text_input: articles['title']})
    
    print('embedding done')
    pd.DataFrame(article_body_embeddings).to_csv(args.export_artile_body_path, index=None)
    pd.DataFrame(article_title_embeddings).to_csv(args.export_artile_title_path, index=None)


parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')

parser.add_argument('-eptp', '--export_prl_title_path', default='content/prl_title_embeddings_use.csv' ,help='この引数の説明（なくてもよい）')    # 必須の引数を追加
parser.add_argument('-eatp', '--export_article_title_path', default='content/article_title_embeddings_use.csv' ,help='この引数の説明（なくてもよい）')
parser.add_argument('-eabp', '--export_artile_body_path', default='content/article_body_embeddings_use.csv' ,help='この引数の説明（なくてもよい）')
parser.add_argument('-pl', '--prl_path', default='data/PRL2011_2017.csv' ,help='この引数の説明（なくてもよい）')
parser.add_argument('-al', '--article_path', default='data/articles.csv' ,help='この引数の説明（なくてもよい）')

args = parser.parse_args()

main(args)
