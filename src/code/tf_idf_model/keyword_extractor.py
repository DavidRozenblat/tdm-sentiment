import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


class TfidfKeywordExtractor:
    def __init__(self, stop_words='english'):
        # Initialize the TfidfVectorizer with given stop words
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)
        self.trained = False

    def train(self, corpus):
        """
        Train the TfidfVectorizer on a provided corpus.
        
        Parameters:
            corpus (list of str): List of documents to train on.
        """
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.trained = True

    def save(self, filepath):
        """
        Save the trained TfidfVectorizer to disk.
        
        Parameters:
            filepath (str): File path to save the model.
        """
        joblib.dump(self.vectorizer, filepath)
        print(f"TF-IDF Vectorizer model saved as '{filepath}'.")

    def load(self, filepath):
        """
        Load a TfidfVectorizer model from disk.
        
        Parameters:
            filepath (str): File path from which to load the model.
        """
        self.vectorizer = joblib.load(filepath)
        self.trained = True
        print(f"TF-IDF Vectorizer model loaded from '{filepath}'.")

    def transform(self, documents):
        """
        Transform new documents into TF-IDF vectors using the trained model.
        
        Parameters:
            documents (list of str): New documents to transform.
        
        Returns:
            Sparse matrix of TF-IDF vectors.
        """
        if not self.trained:
            raise ValueError("The vectorizer has not been trained or loaded yet.")
        return self.vectorizer.transform(documents)

    def extract_top_keywords(self, tfidf_vector, top_n=5):
        """
        Extract the top_n keywords from a single document's TF-IDF vector.
        
        Parameters:
            tfidf_vector: A TF-IDF vector (in sparse format) for one document.
            top_n (int): Number of top keywords to extract.
        
        Returns:
            List of tuples (keyword, score) sorted in descending order by score.
        """
        # Convert the sparse vector to a dense array and flatten it
        dense_vector = tfidf_vector.toarray().flatten()
        # Get indices that would sort the vector in descending order
        top_indices = np.argsort(dense_vector)[::-1][:top_n]
        # Get the feature names from the vectorizer
        feature_names = self.vectorizer.get_feature_names_out()
        return [(feature_names[i], dense_vector[i]) for i in top_indices]

    def get_top_words(self, tfidf_vector, top_n=5):
        """
        Return the top_n words from a single document's TF-IDF vector, without the scores.
        
        Parameters:
            tfidf_vector: A TF-IDF vector (in sparse format) for one document.
            top_n (int): Number of top words to return.
        
        Returns:
            List of top words.
        """
        # Use extract_top_keywords to get the words with scores
        top_keywords = self.extract_top_keywords(tfidf_vector, top_n)
        # Return only the words, without their scores
        return [word for word, score in top_keywords]

    def extract_keywords_from_documents(self, documents, top_n=5):
        """
        Extract top keywords for each document in a list of documents.
        
        Parameters:
            documents (list of str): List of new documents.
            top_n (int): Number of top keywords to extract from each document.
        
        Returns:
            List of lists, where each sublist contains tuples (keyword, score) for a document.
        """
        tfidf_matrix = self.transform(documents)
        keywords_list = []
        for i in range(tfidf_matrix.shape[0]):
            keywords = self.extract_top_keywords(tfidf_matrix[i], top_n=top_n)
            keywords_list.append(keywords)
        return keywords_list

# ----------------- Example Usage -----------------
if __name__ == "__main__":
    # ----- Step 1: Train the TfidfVectorizer on a large corpus -----
    # Replace this list with your large corpus of text documents.
    large_corpus = [
        "This is a sample document about machine learning and data science.",
        "Another example document discussing natural language processing techniques.",
        "More text data helps the model learn vocabulary and contextual information.",
        "This is the first document. It contains text about machine learning and data science.",
        "The second document focuses on natural language processing and machine learning applications.",
        "Text analysis is a part of data science that includes extracting keywords using TF-IDF.",
        "More documents may discuss different topics such as computer vision, AI, and deep learning."
    ]
    
    extractor = TfidfKeywordExtractor()
    extractor.train(large_corpus)
    
    # Save the trained model
    extractor.save('tfidf_vectorizer.pkl')
    
    # Later you can load the model instead of retraining
    extractor.load('tfidf_vectorizer.pkl')
    
    # Step 2: Extract keywords from new documents
    new_documents = [
        "A new document discussing deep learning and neural networks.",
        "This document focuses on data analysis and visualization techniques."
    ]
    
    tfidf_matrix_new = extractor.transform(new_documents)
    
    # For each new document, print the top keywords with scores
    for idx in range(tfidf_matrix_new.shape[0]):
        print(f"\nDocument {idx + 1} top keywords with scores:")
        for keyword, score in extractor.extract_top_keywords(tfidf_matrix_new[idx], top_n=5):
            print(f"  {keyword}: {score:.4f}")
    
    # For one document, print only the top words without scores
    print("\nDocument 1 top words (without scores):")
    top_words = extractor.get_top_words(tfidf_matrix_new[1], top_n=5)
    print(top_words)