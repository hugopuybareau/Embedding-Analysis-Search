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
                min_count=0
            )
        
        elif self.method == 'sbert':
            self.model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
            
        else: raise ValueError('Unknown method -> (word2vec, tfidf, sbert)')
            
        
    def transform(self, texts):
        if self.method == 'tfidf':
            return self.model.transform(texts).toarray()
        
        elif self.method == 'word2vec':
            tokenized_texts = [word_tokenize(text.lower()) for text in texts]
            return np.array([
                np.mean([self.model.wv[word] for word in words if word in self.model.wv] or [np.zeros(100)], axis=0)
                for words in tokenized_texts
            ])
        
        # TF-IDF and Word2Vec won't have the same shape because TF-IDF creates a sparce matrix shaped 
        # (number of documents, number of words)
        # SBERT will be 384 because "paraphrase-MiniLM-L6-v2" is like that. 
        # For example, paraphrase-mpnet-base-v2 would be 768, more accurate but slower. 

        elif self.method == 'sbert':
            return self.model.encode(texts)
        
        else: raise ValueError('Unknown method -> (word2vec, tfidf, sbert)')

    def similarity(self, vec1, vec2):
        return cosine_similarity([vec1], [vec2])[0][0]


        


