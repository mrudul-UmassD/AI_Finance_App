import nltk
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

class SentimentAnalyzer:
    """
    Sentiment analysis model for financial news and texts using VADER and finance-specific lexicons
    """
    
    def __init__(self, model_type='vader'):
        """
        Initialize the sentiment analyzer
        
        Parameters:
        -----------
        model_type : str
            Type of model to use (only 'vader' is supported for lightweight version)
        """
        self.model_type = 'vader'  # Force vader for lightweight version
        
        # Initialize VADER sentiment analyzer
        try:
            self.vader = SentimentIntensityAnalyzer()
        except:
            # If resources not downloaded, download them
            nltk.download('vader_lexicon', quiet=True)
            self.vader = SentimentIntensityAnalyzer()
        
        # Download other NLTK resources if using text preprocessing
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            
        self.stop_words = set(stopwords.words('english'))
        
        # Financial domain-specific lexicon
        self.financial_lexicon = {
            # Positive financial terms
            'growth': 0.7,
            'profit': 0.8,
            'revenue': 0.6,
            'earnings': 0.6,
            'dividend': 0.5,
            'bullish': 0.9,
            'outperform': 0.8,
            'upgrade': 0.7,
            'recovery': 0.6,
            'expansion': 0.6,
            'beat': 0.7,
            'exceeds': 0.7,
            'strong': 0.6,
            'positive': 0.7,
            'upward': 0.6,
            'rise': 0.6,
            'gain': 0.6,
            'improve': 0.6,
            
            # Negative financial terms
            'loss': -0.8,
            'debt': -0.6,
            'bearish': -0.9,
            'downgrade': -0.7,
            'decline': -0.6,
            'recession': -0.8,
            'bankruptcy': -0.9,
            'crash': -0.8,
            'crisis': -0.8,
            'sell-off': -0.7,
            'default': -0.8,
            'miss': -0.7,
            'weak': -0.6,
            'negative': -0.7,
            'downward': -0.6,
            'fall': -0.6,
            'drop': -0.6,
            'worsen': -0.7
        }
    
    def _preprocess_text(self, text):
        """
        Preprocess text for sentiment analysis
        
        Parameters:
        -----------
        text : str
            Raw text
            
        Returns:
        --------
        str
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        
        # Reconstruct text
        preprocessed_text = ' '.join(filtered_tokens)
        
        return preprocessed_text
    
    def analyze(self, text):
        """
        Analyze sentiment of the given text
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        float
            Sentiment score (-1 to 1, where -1 is very negative, 0 is neutral, 1 is very positive)
        """
        # Preprocess text
        preprocessed_text = self._preprocess_text(text)
        
        if not preprocessed_text:
            return 0.0  # Neutral if no text
        
        # Get base sentiment from VADER
        base_sentiment = self._vader_sentiment(preprocessed_text)
        
        # Enhance with finance-specific sentiment
        finance_sentiment = self._financial_sentiment(preprocessed_text)
        
        # Weighted combination
        combined_sentiment = 0.7 * base_sentiment + 0.3 * finance_sentiment
        
        # Ensure within range [-1, 1]
        return max(-1.0, min(1.0, combined_sentiment))
    
    def _vader_sentiment(self, text):
        """
        Get sentiment score using VADER
        
        Parameters:
        -----------
        text : str
            Preprocessed text
            
        Returns:
        --------
        float
            Sentiment score (-1 to 1)
        """
        # Get sentiment scores
        scores = self.vader.polarity_scores(text)
        
        # Return compound score (normalized from -1 to 1)
        return scores['compound']
    
    def _financial_sentiment(self, text):
        """
        Get finance-specific sentiment using the custom lexicon
        
        Parameters:
        -----------
        text : str
            Preprocessed text
            
        Returns:
        --------
        float
            Finance-specific sentiment score (-1 to 1)
        """
        words = text.split()
        total_score = 0.0
        matched_words = 0
        
        for word in words:
            if word in self.financial_lexicon:
                total_score += self.financial_lexicon[word]
                matched_words += 1
        
        # If we found any financial terms, return the average score
        if matched_words > 0:
            return total_score / matched_words
        else:
            return 0.0  # Neutral if no financial terms found
    
    def analyze_multiple(self, texts):
        """
        Analyze sentiment of multiple texts
        
        Parameters:
        -----------
        texts : list
            List of texts to analyze
            
        Returns:
        --------
        list
            List of sentiment scores
        """
        return [self.analyze(text) for text in texts]
    
    def get_sentiment_label(self, score):
        """
        Convert sentiment score to human-readable label
        
        Parameters:
        -----------
        score : float
            Sentiment score (-1 to 1)
            
        Returns:
        --------
        str
            Sentiment label
        """
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    
    def get_finance_specific_sentiment(self, text):
        """
        Get finance-specific sentiment with additional financial context
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        dict
            Dictionary with scores and context
        """
        # List of positive and negative financial terms for counting
        positive_terms = [term for term, score in self.financial_lexicon.items() if score > 0]
        negative_terms = [term for term, score in self.financial_lexicon.items() if score < 0]
        
        # Preprocess text
        preprocessed_text = self._preprocess_text(text)
        
        # Get base sentiment
        sentiment_score = self._vader_sentiment(preprocessed_text)
        
        # Get finance sentiment
        finance_sentiment = self._financial_sentiment(preprocessed_text)
        
        # Count finance-specific terms
        words = preprocessed_text.split()
        positive_count = sum(1 for word in words if word in positive_terms)
        negative_count = sum(1 for word in words if word in negative_terms)
        
        # Calculate finance bias factor
        if positive_count + negative_count > 0:
            finance_bias = (positive_count - negative_count) / (positive_count + negative_count)
        else:
            finance_bias = 0
        
        # Calculate adjusted score
        adjusted_score = 0.7 * sentiment_score + 0.3 * finance_sentiment
        
        # Clip to range [-1, 1]
        adjusted_score = max(min(adjusted_score, 1.0), -1.0)
        
        return {
            'base_score': sentiment_score,
            'finance_score': finance_sentiment,
            'finance_bias': finance_bias,
            'adjusted_score': adjusted_score,
            'positive_terms': positive_count,
            'negative_terms': negative_count,
            'label': self.get_sentiment_label(adjusted_score)
        } 