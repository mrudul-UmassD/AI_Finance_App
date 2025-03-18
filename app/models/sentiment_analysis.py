import nltk
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import re

class SentimentAnalyzer:
    """
    Sentiment analysis model for financial news and texts
    """
    
    def __init__(self, model_type='vader'):
        """
        Initialize the sentiment analyzer
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('vader', 'transformers', or 'ensemble')
        """
        self.model_type = model_type
        
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
        
        # Initialize transformer model for more advanced sentiment analysis
        if model_type in ['transformers', 'ensemble']:
            try:
                # Financial domain-specific sentiment model
                model_name = "ProsusAI/finbert"
                self.transformer_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.transformer_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.transformer = pipeline("sentiment-analysis", model=self.transformer_model, tokenizer=self.transformer_tokenizer)
            except:
                # Fallback to generic sentiment model
                self.transformer = pipeline("sentiment-analysis")
    
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
        
        if self.model_type == 'vader':
            return self._vader_sentiment(preprocessed_text)
        elif self.model_type == 'transformers':
            return self._transformer_sentiment(preprocessed_text)
        elif self.model_type == 'ensemble':
            return self._ensemble_sentiment(preprocessed_text)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
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
    
    def _transformer_sentiment(self, text):
        """
        Get sentiment score using transformer model
        
        Parameters:
        -----------
        text : str
            Preprocessed text
            
        Returns:
        --------
        float
            Sentiment score (-1 to 1)
        """
        # Handle text that is too long by chunking
        max_length = 512
        if len(text) > max_length:
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            scores = []
            
            for chunk in chunks:
                result = self.transformer(chunk)
                
                # Map label to score
                if result[0]['label'] == 'POSITIVE':
                    scores.append(result[0]['score'])
                elif result[0]['label'] == 'NEGATIVE':
                    scores.append(-result[0]['score'])
                else:  # NEUTRAL
                    scores.append(0.0)
            
            # Average the scores
            return np.mean(scores)
        else:
            result = self.transformer(text)
            
            # Map label to score
            if result[0]['label'] == 'POSITIVE':
                return result[0]['score']
            elif result[0]['label'] == 'NEGATIVE':
                return -result[0]['score']
            else:  # NEUTRAL
                return 0.0
    
    def _ensemble_sentiment(self, text):
        """
        Get sentiment score using ensemble of models
        
        Parameters:
        -----------
        text : str
            Preprocessed text
            
        Returns:
        --------
        float
            Sentiment score (-1 to 1)
        """
        # Get scores from both models
        vader_score = self._vader_sentiment(text)
        transformer_score = self._transformer_sentiment(text)
        
        # Weighted average (giving more weight to the transformer model)
        return 0.3 * vader_score + 0.7 * transformer_score
    
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
        # List of positive financial terms
        positive_terms = [
            'growth', 'profit', 'revenue', 'earnings', 'dividend', 'bullish',
            'outperform', 'upgrade', 'recovery', 'expansion', 'beat', 'exceeds',
            'strong', 'positive', 'upward', 'rise', 'gain', 'improve'
        ]
        
        # List of negative financial terms
        negative_terms = [
            'loss', 'debt', 'bearish', 'downgrade', 'decline', 'recession',
            'bankruptcy', 'crash', 'crisis', 'sell-off', 'default', 'miss',
            'weak', 'negative', 'downward', 'fall', 'drop', 'worsen'
        ]
        
        # Preprocess text
        preprocessed_text = self._preprocess_text(text)
        
        # Get base sentiment
        sentiment_score = self.analyze(text)
        
        # Count finance-specific terms
        words = preprocessed_text.split()
        positive_count = sum(1 for word in words if word in positive_terms)
        negative_count = sum(1 for word in words if word in negative_terms)
        
        # Calculate finance bias factor
        if positive_count + negative_count > 0:
            finance_bias = (positive_count - negative_count) / (positive_count + negative_count)
        else:
            finance_bias = 0
        
        # Adjust sentiment with finance bias
        adjusted_score = 0.7 * sentiment_score + 0.3 * finance_bias
        
        # Clip to range [-1, 1]
        adjusted_score = max(min(adjusted_score, 1.0), -1.0)
        
        return {
            'base_score': sentiment_score,
            'finance_bias': finance_bias,
            'adjusted_score': adjusted_score,
            'positive_terms': positive_count,
            'negative_terms': negative_count,
            'label': self.get_sentiment_label(adjusted_score)
        } 