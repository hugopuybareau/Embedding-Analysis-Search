import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize

# import nltk
# nltk.download('punkt_tab')

class EmbeddingModel:
    def __init__(self, method):
        self.method = method
        self.model = None
    
    def fit(self, texts):
        if self.method == 'tfidf':
            self.model = TfidfVectorizer()
            self.model.fit(texts)

        elif self.method == 'word2vec':
            tokenized_texts = [word_tokenize(text.lower()) for text in texts]
            # print(tokenized_texts)
            self.model = Word2Vec(
                sentences=tokenized_texts,
                vector_size=100,
                window=5,
                min_count=1
            )
        
        elif self.method == 'sbert':
            self.model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        else:
            raise ValueError('Unknown method -> (word2vec, tfidf, sbert)')
        


