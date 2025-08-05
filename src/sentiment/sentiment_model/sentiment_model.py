from pathlib import Path
from typing import List
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoConfig,
    pipeline
)


class TextAnalysis:
    """
    A class for initializing and using a BERT-based sentiment analysis pipeline.

    Attributes:
        local_model_path (Path): Path to the local directory containing the model files.
        load_weights_only (bool): Whether to load only the model weights securely.
        device (int): Device index for running inference. Defaults to CPU (-1), uses GPU if available.
        sentiment_pipeline (pipeline): The initialized Hugging Face sentiment analysis pipeline.
    """
    
    def __init__(self, 
                 local_model_path: Path, 
                 load_weights_only: bool = True,
                 device: int = None):
        """
        Initialize the TextAnalysis class with a local BERT-based sentiment analyzer.

        Args:
            local_model_path (str): The local directory containing the saved model.
            load_weights_only (bool): If True, loads only model weights securely.
            device (int, optional): The device index for model inference. 
                -1 = CPU, 0 or higher = GPU index. If None, tries GPU if available, else CPU.
        """
        self.local_model_path = local_model_path
        self.load_weights_only = load_weights_only
        self.sentiment_pipeline = None

        # Determine device automatically if not provided
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device

        self._load_model()
        
        
    def get_pipeline(self):
        """
        Retrieve the initialized sentiment analysis pipeline.

        Returns:
            pipeline: The Hugging Face sentiment analysis pipeline.
        
        Raises:
            RuntimeError: If the pipeline has not been initialized.
        """
        if self.sentiment_pipeline is None:
            raise RuntimeError("Sentiment pipeline has not been initialized.")
        return self.sentiment_pipeline


    def _load_model(self):
        """
        Load the model and tokenizer, and initialize the sentiment pipeline.
        """
        try:
            if self.load_weights_only:
                # Load model configuration and weights from the given path
                config = AutoConfig.from_pretrained(str(self.local_model_path))
                model = AutoModelForSequenceClassification.from_pretrained(
                    str(self.local_model_path), config=config
                )
            else:
                # Load the full model directly from the local directory
                model = AutoModelForSequenceClassification.from_pretrained(
                    str(self.local_model_path)
                )

            tokenizer = AutoTokenizer.from_pretrained(str(self.local_model_path))

            self.sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model=model,
                tokenizer=tokenizer,
                device=self.device
            )

            print(f"BERT model loaded successfully from '{self.local_model_path}' on device {self.device}.")
        except Exception as e:
            print(f"Error loading BERT model from '{self.local_model_path}': {e}")
            raise


    def txt_sentiment_dict(self, my_txt: str) -> dict:
        """
        Get a dictionary of all sentiment labels and their probabilities for the provided text.
        Parameters:
            my_txt (str): Input text to analyze.
        Returns:
            dict: A dictionary mapping each sentiment label (converted to lowercase) to its probability.
                  For example: {'negative': 0.05, 'neutral': 0.10, 'positive': 0.85}
        """
        # Ensure the input is shorter than 512 characters
        if len(my_txt) > 512:
            print(f'text Truncate, len text is{len(my_txt)}')
            my_txt = my_txt[:511]
        pipeline_result = self.get_pipeline()(my_txt, return_all_scores=True)
        # Assume the first element of the result list contains the sentiment scores
        scores = {entry['label'].lower(): round(entry['score'], 4) for entry in pipeline_result[0]}
        return scores



    def txt_score(self, my_txt: str) -> float:
        """
        Calculate the sentiment score for a given text.
        
        The score is computed as (positive - negative) / (positive + negative).
        Uses get_sentiment_dict() as a helper function.
        
        Parameters:
            my_txt (str): Input text to analyze.
        
        Returns:
            float: The computed sentiment score or 0 if the sum of positive and negative scores is 0.
        """
        if not isinstance(my_txt, str):
            return None
        # Truncate the text if it exceeds 512 characters (or tokens as needed)
        if len(my_txt) > 512:
            my_txt = my_txt[:511]

        scores = self.get_sentiment_dict(my_txt)
        positive = scores.get('positive', 0)
        negative = scores.get('negative', 0)
        
        # Guard against division by zero
        if positive + negative == 0:
            return 0
        sentiment_score = (positive - negative) / (positive + negative)
        return sentiment_score


    def get_article_sentiment_helper(self, inp) -> float:
        """
        Calculate the sentiment score of the input text(s) using a weighted average.
        
        Parameters:
            inp (str or list): A single string or a list of strings representing the text(s).
        
        Returns:
            float: The average sentiment score of the input text(s). Returns 0 for empty input.
        """
        if isinstance(inp, str):
            inp = [inp]  # Convert a single string to a list
        elif not isinstance(inp, list) or not all(isinstance(text, str) for text in inp):
            print(f"Input must be a string or a list of strings but got: {inp} with type: {type(inp)}")
            return 0
        
        word_count = sum(len(text.split()) for text in inp)
        if word_count == 0:
            return 0

        try:
            sentiment_scores = [self.txt_score(text) for text in inp if isinstance(text, str)]
            return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        except Exception as e:
            print(f"Error processing input {inp}: {e}")
            return 0

    def get_article_sentiment(self, row, weights=None) -> float:
        """
        Calculate the weighted sentiment score for an article.
        
        Parameters:
            row (dict-like): A row with keys 'title', 'subtitle', and 'body_paragraphs'.
            weights (dict): Optional dictionary with keys 'title', 'subtitle', 'body_text'
                            specifying their weights.
        
        Returns:
            float: The weighted sentiment score.
        """
        try:
            title_score = self.get_article_sentiment_helper(row['title']) if row.get('title') else None
            # subtitle_score = self.get_article_sentiment_helper(row['subtitle']) if row.get('subtitle') else None
            body_text_score = self.get_article_sentiment_helper(row['body_paragraphs']) if row.get('body_paragraphs') else None
            
            # Default weights if not provided
            if weights is None:
                weights = {'title': 0.2, 'subtitle': 0.0, 'body_text': 0.8}
            
            weights = {
                'title': weights['title'] if title_score is not None else 0,
                # 'subtitle': weights['subtitle'] if subtitle_score is not None else 0,
                'body_text': weights['body_text'] if body_text_score is not None else 0
            }
            total_weight = sum(weights.values())
            if total_weight == 0:
                return 0
            normalized_weights = {key: value / total_weight for key, value in weights.items()}
            
            score = (
                (title_score or 0) * normalized_weights.get('title', 0) +
                # (subtitle_score or 0) * normalized_weights.get('subtitle', 0) +
                (body_text_score or 0) * normalized_weights.get('body_text', 0)
            )
            return score
        except Exception as e:
            print(f"Error processing row {row}: {e}")
            return None

    def analyze_article_sentiment(self, texts: List[str], method: str = 'bert') -> float:
        """
        Analyzes the sentiment of an article represented as a list of texts using the BERT-based model.
        
        Parameters:
            texts (List[str]): A list containing multiple sentences or paragraphs.
            method (str): The sentiment analysis method to use. Only 'bert' is supported.
        
        Returns:
            float: The overall sentiment score of the article, between -1 and 1.
        """
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("Input must be a list of strings.")
        
        method = method.lower().strip()
        if method == 'bert':
            return self.get_article_sentiment_helper(texts)
        else:
            raise ValueError("Invalid method specified. Only 'bert' is supported.")


