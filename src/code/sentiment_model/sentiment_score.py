from textblob import TextBlob
from transformers import pipeline
import torch


class TextAnalysis:
    def __init__(self, 
                 model_name="distilbert-base-uncased-finetuned-sst-2-english", 
                 device=-1):
        """
        Initializes the TextAnalysis class with both TextBlob and BERT-based sentiment analyzers.

        Parameters:
        - model_name (str): The name of the pre-trained BERT model to use for sentiment analysis.
        - device (int): Device to run the model on. 
                        -1 for CPU, 0 for GPU 0, 1 for GPU 1, etc.
        """
        self.model_name = model_name
        self.device = device
        self.sentiment_pipeline = None

        # Initialize the BERT-based sentiment analysis pipeline
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model=self.model_name, 
                device=self.device
            )
            print(f"BERT model '{self.model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading BERT model '{self.model_name}': {e}")


    def text_blob_sentiment(self, texts):
        """
        Get the sentiment of a list of texts using TextBlob.

        Args:
            texts (list): A list containing multiple sentences or paragraphs.

        Returns:
            float: The overall sentiment of the text, a number between -1 and 1.
                   -1 indicates very negative sentiment.
                   +1 indicates very positive sentiment.
                    0 indicates neutral sentiment.
        """
        if not isinstance(texts, list):
            raise ValueError("Input must be a list of strings.")

        # Concatenate all texts into a single string
        concatenated_text = ' '.join(texts)
        
        # Analyze sentiment using TextBlob
        blob = TextBlob(concatenated_text)
        total_sentiment = blob.sentiment.polarity  # Range from -1 to 1

        return total_sentiment

    def truncate_text(self, text, max_length=512):
        """
        Truncates the text to a maximum number of characters.

        Args:
            text (str): The text to truncate.
            max_length (int): The maximum allowed number of characters.

        Returns:
            str: The truncated text.
        """
        if len(text) > max_length:
            truncated = text[:max_length]
            print(f"Truncated text from {len(text)} to {len(truncated)} characters.")
            return truncated
        return text

    
    def bert_sentiment(self, texts):
        """
        Get the sentiment of a list of texts using a BERT-based model.

        Args:
            texts (list): A list containing multiple sentences or paragraphs.

        Returns:
            float: The mean sentiment score of the texts, a number between -1 and 1.
                -1 indicates very negative sentiment.
                +1 indicates very positive sentiment.
                0 indicates neutral sentiment.
        """
        if not isinstance(texts, list):
            raise ValueError("Input must be a list of strings.")

        if not self.sentiment_pipeline:
            print("BERT sentiment pipeline is not initialized.")
            return 0.0  # Neutral sentiment as default

        try:
            sentiment_scores = []
            total_tokens = 0  # Total number of tokens across all texts

            for text in texts:
                # Calculate the weighted sentiment for each text
                text_sentiment, text_token_count = self.calculate_weighted_sentiment(text)
                sentiment_scores.append((text_sentiment, text_token_count))
                total_tokens += text_token_count

            # Calculate the overall weighted average sentiment across all texts
            if sentiment_scores and total_tokens > 0:
                overall_weighted_sum = sum(s * t for s, t in sentiment_scores)
                mean_sentiment = overall_weighted_sum / total_tokens
            else:
                mean_sentiment = 0.0  # Neutral sentiment if no scores are available

            return mean_sentiment

        except Exception as e:
            print(f"Error during BERT sentiment analysis: {e}")
            return 0.0  # Neutral sentiment as default
        

    def calculate_weighted_sentiment(self, text):
        """
        Calculate the weighted sentiment score for a single text.

        Args:
            text (str): The text to analyze.

        Returns:
            tuple: A tuple containing the weighted sentiment score and the total token count.
        """
        # Tokenize the text without truncation
        encoded_input = self.sentiment_pipeline.tokenizer(
            text,
            truncation=False,
            return_tensors='pt'
        )

        input_ids = encoded_input['input_ids']
        num_tokens = input_ids.size(1)
        max_length = self.sentiment_pipeline.model.config.max_position_embeddings

        # Split into chunks if necessary
        if num_tokens > max_length:
            # Flatten the token IDs and split into chunks
            tokens = input_ids.squeeze().tolist()
            chunks = [
                tokens[i:i + max_length]
                for i in range(0, len(tokens), max_length)
            ]
        else:
            chunks = [input_ids.squeeze().tolist()]

        # Analyze sentiment for each chunk
        chunk_scores = []
        chunk_token_counts = []

        for chunk_tokens in chunks:
            # Decode tokens back to text
            chunk_text = self.sentiment_pipeline.tokenizer.decode(
                chunk_tokens, skip_special_tokens=True
            )
            # Get sentiment prediction for the chunk
            result = self.sentiment_pipeline(chunk_text)[0]
            label = result.get('label', 'NEUTRAL').upper()
            score = result.get('score', 0.0)

            if label == 'POSITIVE':
                sentiment_score = score  # Positive sentiment
            elif label == 'NEGATIVE':
                sentiment_score = -score  # Negative sentiment
            else:
                sentiment_score = 0.0  # Neutral or undefined sentiment

            # Append the sentiment score and token count
            chunk_scores.append(sentiment_score)
            token_count = len(chunk_tokens)
            chunk_token_counts.append(token_count)

        # Calculate the weighted average sentiment for the text
        if chunk_scores and sum(chunk_token_counts) > 0:
            weighted_sum = sum(s * t for s, t in zip(chunk_scores, chunk_token_counts))
            total_tokens = sum(chunk_token_counts)
            text_sentiment = weighted_sum / total_tokens
        else:
            text_sentiment = 0.0  # Neutral sentiment if no scores are available
            total_tokens = 0

        return text_sentiment, total_tokens



    def analyze_article_sentiment(self, texts, method='bert'):
        """
        Analyzes the sentiment of an article represented as a list of texts.

        Args:
            texts (list): A list containing multiple sentences or paragraphs.
            method (str): The sentiment analysis method to use ('textblob' or 'bert').

        Returns:
            float: The overall sentiment score of the article, between -1 and 1.
        """
        if method.lower() == 'textblob':
            return self.text_blob_sentiment(texts)
        elif method.lower() == 'bert':
            return self.bert_sentiment(texts)
        else:
            raise ValueError("Invalid method specified. Choose 'textblob' or 'bert'.")
