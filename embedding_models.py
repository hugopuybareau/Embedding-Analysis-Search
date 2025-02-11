import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

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
    
    def visualize_embeddings(self, texts, labels, method):
        embeddings = self.transform(texts)
        if method == 'pca':
            reducer = PCA(n_components=2)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=3, random_state=0)
            # perplexity must be less than n_samples
        elif method == 'tsne scaled':
            scaler = StandardScaler()
            embeddings = StandardScaler().fit_transform(embeddings)
            reducer = TSNE(n_components=2, perplexity=2, random_state=0)
        else: raise ValueError('Unknown method -> (pca, tsne, tsne scaled)')

        reduced_embeddings = reducer.fit_transform(embeddings)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1],
            hue=labels,
            palette="coolwarm",
            s=100
        )
        for i, text in enumerate(texts):
            plt.annotate(
                text[:15],
                (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                 fontsize=9,
                 alpha=0.7
            )
        plt.title(f"Embedding Projection using {method.upper()}")
        plt.grid()
        plt.show()

    def index_texts(self, texts): # I create a memory for the texts' embeddings before the search thing
        embeddings = self.transform(texts)
        self.texts_embeddings = {i: (doc, embeddings[i]) for i, doc in enumerate(texts)}
        print(f"Indexed {len(texts)} texts.")

    def search_similar(self, query, how_much_results):
        if not self.texts_embeddings : raise ValueError('Call index_texts() first.')

        query_embedding = self.transform([query])[0]
        texts_embeddings_arr = np.array([emb[1] for emb in self.texts_embeddings.values()])

        similarities = cosine_similarity([query_embedding], texts_embeddings_arr)[0]
        top_indices = np.argsort(similarities)[::-1][:how_much_results]

        return [(self.texts_embeddings[i][0], similarities[i]) for i in top_indices]





        


