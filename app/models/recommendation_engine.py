import numpy as np
from datetime import datetime, timedelta

class RecommendationEngine:
    """
    Generates investment recommendations based on technical analysis, sentiment, and user profile
    """
    
    def __init__(self):
        """Initialize the recommendation engine"""
        self.risk_profiles = {
            "Very Low": {
                "technical_weight": 0.3,
                "sentiment_weight": 0.1,
                "prediction_weight": 0.2,
                "fundamental_weight": 0.4
            },
            "Low": {
                "technical_weight": 0.3,
                "sentiment_weight": 0.15,
                "prediction_weight": 0.25,
                "fundamental_weight": 0.3
            },
            "Medium": {
                "technical_weight": 0.3,
                "sentiment_weight": 0.2,
                "prediction_weight": 0.3,
                "fundamental_weight": 0.2
            },
            "High": {
                "technical_weight": 0.25,
                "sentiment_weight": 0.25,
                "prediction_weight": 0.35,
                "fundamental_weight": 0.15
            },
            "Very High": {
                "technical_weight": 0.2,
                "sentiment_weight": 0.3,
                "prediction_weight": 0.4,
                "fundamental_weight": 0.1
            }
        }
    
    def get_recommendations(self, ticker, risk_profile="Medium", data=None, sentiment=None):
        """
        Generate investment recommendations for a stock based on risk profile
        
        Parameters:
        -----------
        ticker : str
            Stock symbol
        risk_profile : str
            User's risk profile ('Very Low', 'Low', 'Medium', 'High', 'Very High')
        data : pandas.DataFrame, optional
            Stock data if already fetched
        sentiment : float, optional
            Sentiment score if already calculated
            
        Returns:
        --------
        list
            List of recommendation dictionaries
        """
        # If we don't have a valid risk profile, use Medium
        if risk_profile not in self.risk_profiles:
            risk_profile = "Medium"
        
        # Generate mock recommendations for demonstration
        mock_recs = self._get_mock_recommendations(ticker, risk_profile)
        return mock_recs
    
    def _get_mock_recommendations(self, ticker, risk_profile):
        """Generate mock recommendations for demonstration purposes"""
        recommendations = []
        
        if risk_profile in ["Very Low", "Low"]:
            recommendations.append({
                "title": "Conservative Position",
                "description": f"Consider a small position in {ticker} as part of a diversified portfolio. Limit exposure to no more than 2-3% of your total investments.",
                "confidence": 0.85
            })
            
            recommendations.append({
                "title": "Dollar-Cost Averaging",
                "description": f"Rather than investing a lump sum in {ticker}, consider buying in smaller amounts over time to reduce the impact of volatility.",
                "confidence": 0.9
            })
            
            recommendations.append({
                "title": "Protective Options Strategy",
                "description": f"If investing in {ticker}, consider using protective puts to limit downside risk.",
                "confidence": 0.7
            })
            
        elif risk_profile == "Medium":
            recommendations.append({
                "title": "Balanced Position",
                "description": f"Consider a moderate position in {ticker} as part of a balanced portfolio. Limit exposure to 3-5% of your total investments.",
                "confidence": 0.8
            })
            
            recommendations.append({
                "title": "Technical Entry Points",
                "description": f"Look for technical support levels to initiate positions in {ticker}, with clear stop-loss levels to manage risk.",
                "confidence": 0.75
            })
            
            recommendations.append({
                "title": "Sector Allocation",
                "description": f"Ensure your investment in {ticker} doesn't overweight your portfolio in its sector.",
                "confidence": 0.85
            })
            
        else:  # High or Very High
            recommendations.append({
                "title": "Growth Opportunity",
                "description": f"Consider a significant position in {ticker} for potential growth. Could allocate 5-10% of portfolio depending on conviction.",
                "confidence": 0.7
            })
            
            recommendations.append({
                "title": "Momentum Strategy",
                "description": f"Use technical momentum indicators to time entries and exits for {ticker}, looking to capture shorter-term price movements.",
                "confidence": 0.65
            })
            
            recommendations.append({
                "title": "Options Strategy",
                "description": f"Consider using call options on {ticker} to leverage your position for potentially higher returns (with higher risk).",
                "confidence": 0.6
            })
        
        # Add a common recommendation for all risk profiles
        recommendations.append({
            "title": "Stay Informed",
            "description": f"Monitor news, earnings reports, and sector trends that could impact {ticker}.",
            "confidence": 0.95
        })
        
        return recommendations
    
    def _analyze_fundamentals(self, stock_info):
        """
        Analyze fundamental metrics
        
        Parameters:
        -----------
        stock_info : dict
            Dictionary with fundamental metrics
            
        Returns:
        --------
        float
            Fundamental score (-1 to 1)
        """
        score = 0.0
        count = 0
        
        # PE ratio analysis
        if 'forwardPE' in stock_info and stock_info['forwardPE'] is not None:
            pe = stock_info['forwardPE']
            if pe < 15:  # Below average PE
                score += 0.5
            elif pe > 30:  # High PE
                score -= 0.3
            else:  # Average PE
                score += 0.1
            count += 1
        
        # Dividend yield analysis
        if 'dividendYield' in stock_info and stock_info['dividendYield'] is not None:
            div_yield = stock_info['dividendYield']
            if div_yield > 4:  # High yield
                score += 0.7
            elif div_yield > 2:  # Average yield
                score += 0.3
            count += 1
        
        # Beta analysis
        if 'beta' in stock_info and stock_info['beta'] is not None:
            beta = stock_info['beta']
            if beta < 0.8:  # Low volatility
                score += 0.4
            elif beta > 1.5:  # High volatility
                score -= 0.2
            count += 1
        
        # PEG ratio analysis
        if 'pegRatio' in stock_info and stock_info['pegRatio'] is not None:
            peg = stock_info['pegRatio']
            if peg < 1:  # Potentially undervalued
                score += 0.6
            elif peg > 2:  # Potentially overvalued
                score -= 0.4
            count += 1
        
        # Calculate average score
        if count > 0:
            return score / count
        else:
            return 0.0
    
    def _get_technical_score(self, signals, patterns):
        """
        Calculate overall technical score
        
        Parameters:
        -----------
        signals : dict
            Dictionary with technical signals
        patterns : dict
            Dictionary with chart patterns
            
        Returns:
        --------
        float
            Technical score (-1 to 1)
        """
        # Start with overall signal
        if 'OVERALL' in signals:
            if signals['OVERALL']['signal'] == 'BUY':
                score = signals['OVERALL']['strength']
            elif signals['OVERALL']['signal'] == 'SELL':
                score = -signals['OVERALL']['strength']
            else:
                score = 0.0
        else:
            # Calculate from individual signals
            buy_count = sum(1 for s in signals.values() if s['signal'] == 'BUY')
            sell_count = sum(1 for s in signals.values() if s['signal'] == 'SELL')
            neutral_count = sum(1 for s in signals.values() if s['signal'] == 'NEUTRAL')
            
            buy_strength = sum(s['strength'] for s in signals.values() if s['signal'] == 'BUY')
            sell_strength = sum(s['strength'] for s in signals.values() if s['signal'] == 'SELL')
            
            total_count = buy_count + sell_count + neutral_count
            if total_count > 0:
                score = (buy_strength - sell_strength) / total_count
            else:
                score = 0.0
        
        # Adjust score based on patterns
        for pattern, details in patterns.items():
            if 'Bullish' in details['type']:
                score += 0.1 * details['confidence']
            elif 'Bearish' in details['type']:
                score -= 0.1 * details['confidence']
        
        # Normalize to [-1, 1] range
        return max(min(score, 1.0), -1.0)
    
    def _classify_recommendation(self, score):
        """
        Classify recommendation based on score
        
        Parameters:
        -----------
        score : float
            Overall score (-1 to 1)
            
        Returns:
        --------
        str
            Recommendation type
        """
        if score >= 0.6:
            return "Strong Buy"
        elif score >= 0.2:
            return "Buy"
        elif score <= -0.6:
            return "Strong Sell"
        elif score <= -0.2:
            return "Sell"
        else:
            return "Hold"
    
    def _generate_description(self, ticker, rec_type, signals, patterns, stock_info):
        """
        Generate recommendation description
        
        Parameters:
        -----------
        ticker : str
            Stock symbol
        rec_type : str
            Recommendation type
        signals : dict
            Technical signals
        patterns : dict
            Chart patterns
        stock_info : dict
            Stock fundamental info
            
        Returns:
        --------
        str
            Recommendation description
        """
        # Start with recommendation
        if rec_type == "Strong Buy":
            desc = f"Our analysis indicates very positive prospects for {ticker}. "
        elif rec_type == "Buy":
            desc = f"Our analysis suggests favorable conditions for {ticker}. "
        elif rec_type == "Hold":
            desc = f"Our analysis suggests maintaining current positions in {ticker}. "
        elif rec_type == "Sell":
            desc = f"Our analysis suggests reducing positions in {ticker}. "
        elif rec_type == "Strong Sell":
            desc = f"Our analysis indicates significant downside risk for {ticker}. "
        
        # Add technical signal info
        if 'OVERALL' in signals:
            if signals['OVERALL']['signal'] == 'BUY' and signals['OVERALL']['strength'] > 0.6:
                desc += "Technical indicators show strongly bullish signals. "
            elif signals['OVERALL']['signal'] == 'BUY':
                desc += "Technical indicators show moderately bullish signals. "
            elif signals['OVERALL']['signal'] == 'SELL' and signals['OVERALL']['strength'] > 0.6:
                desc += "Technical indicators show strongly bearish signals. "
            elif signals['OVERALL']['signal'] == 'SELL':
                desc += "Technical indicators show moderately bearish signals. "
            else:
                desc += "Technical indicators show mixed or neutral signals. "
        
        # Add pattern information if available
        if patterns:
            pattern_names = list(patterns.keys())
            if pattern_names:
                desc += f"We've identified a {pattern_names[0]} pattern "
                if 'Bullish' in patterns[pattern_names[0]]['type']:
                    desc += "which typically signals upward price movement. "
                elif 'Bearish' in patterns[pattern_names[0]]['type']:
                    desc += "which typically signals downward price movement. "
                else:
                    desc += "which should be monitored closely. "
        
        # Add fundamental info if available
        if 'sector' in stock_info and stock_info['sector']:
            desc += f"As a {stock_info['sector']} company, "
            
            if 'forwardPE' in stock_info and stock_info['forwardPE'] is not None:
                pe = stock_info['forwardPE']
                if pe < 15:
                    desc += f"{ticker} is trading at an attractive forward P/E ratio of {pe:.1f}. "
                elif pe > 30:
                    desc += f"{ticker} is trading at a premium forward P/E ratio of {pe:.1f}. "
                else:
                    desc += f"{ticker} is trading at a reasonable forward P/E ratio of {pe:.1f}. "
            
            if 'dividendYield' in stock_info and stock_info['dividendYield'] is not None:
                div_yield = stock_info['dividendYield']
                if div_yield > 0:
                    desc += f"It offers a dividend yield of {div_yield:.1f}%. "
        
        return desc
    
    def _get_diversification_recommendation(self, ticker, risk_profile):
        """
        Generate portfolio diversification recommendation
        
        Parameters:
        -----------
        ticker : str
            Stock symbol
        risk_profile : str
            Risk profile
            
        Returns:
        --------
        dict
            Diversification recommendation
        """
        # Get related stocks
        related_stocks = get_related_stocks(ticker)
        
        if not related_stocks:
            related_stocks = ["Diversified ETFs", "Sector-specific funds", "Bonds"]
        
        # Generate recommendation
        if risk_profile in ['Very Low', 'Low']:
            title = "Conservative Diversification Strategy"
            desc = (f"To complement your {ticker} position with reduced risk, consider allocating to more "
                   f"stable assets. We recommend adding blue-chip stocks, bonds, and dividend ETFs. "
                   f"Related securities to explore: {', '.join(related_stocks[:3])}.")
        elif risk_profile == 'Medium':
            title = "Balanced Diversification Strategy"
            desc = (f"To build a balanced portfolio around your {ticker} position, "
                   f"consider a mix of growth and value stocks across different sectors. "
                   f"Related securities to explore: {', '.join(related_stocks[:3])}.")
        else:
            title = "Growth-Focused Diversification Strategy"
            desc = (f"To maximize growth potential alongside your {ticker} position, "
                   f"look for complementary growth stocks in different sectors. "
                   f"Consider adding exposure to emerging technologies and high-potential industries. "
                   f"Related securities to explore: {', '.join(related_stocks[:3])}.")
        
        return {
            'title': title,
            'description': desc,
            'confidence': 0.75
        }
    
    def _get_generic_recommendations(self, risk_profile):
        """
        Generate generic recommendations when specific analysis fails
        
        Parameters:
        -----------
        risk_profile : str
            Risk profile
            
        Returns:
        --------
        list
            List of generic recommendation dictionaries
        """
        if risk_profile not in self.strategies:
            risk_profile = 'Medium'  # Default to medium if invalid profile
        
        recommendations = []
        
        # Add both strategies for this risk profile
        for strategy in self.strategies[risk_profile]:
            strategy_copy = strategy.copy()
            strategy_copy['confidence'] = 0.7
            recommendations.append(strategy_copy)
        
        # Add general market recommendation
        recommendations.append({
            'title': 'Market Timing Strategy',
            'description': ('Consider a dollar-cost averaging approach to enter the market gradually. '
                           'This helps mitigate the risk of market timing and reduces the impact of volatility.'),
            'confidence': 0.8
        })
        
        return recommendations
    
    def _get_mock_sentiment_score(self, ticker):
        """
        Generate mock sentiment score for demo purposes
        
        Parameters:
        -----------
        ticker : str
            Stock symbol
            
        Returns:
        --------
        float
            Sentiment score (-1 to 1)
        """
        # Some popular stocks with sentiment bias for demo
        sentiments = {
            'AAPL': 0.6,  # Generally positive
            'MSFT': 0.5,  # Generally positive
            'GOOGL': 0.4,  # Positive
            'AMZN': 0.3,  # Somewhat positive
            'META': 0.1,  # Neutral to slightly positive
            'NFLX': 0.2,  # Somewhat positive
            'TSLA': 0.0,  # Highly divisive, neutral overall
            'NVDA': 0.7,  # Very positive
            'AMD': 0.4,  # Positive
            'INTC': -0.2,  # Somewhat negative
            'IBM': -0.1,  # Slightly negative
            'GE': -0.3,  # Somewhat negative
            'F': -0.1,  # Slightly negative
            'GM': -0.2,  # Somewhat negative
            'BAC': 0.1,  # Neutral to slightly positive
            'JPM': 0.2,  # Somewhat positive
            'WMT': 0.1,  # Neutral to slightly positive
            'TGT': 0.0,  # Neutral
            'KO': 0.3,  # Somewhat positive
            'PEP': 0.2   # Somewhat positive
        }
        
        # Return sentiment if available, otherwise generate a random sentiment
        if ticker in sentiments:
            return sentiments[ticker]
        else:
            # Generate a random sentiment between -0.5 and 0.5
            return np.random.uniform(-0.5, 0.5)
    
    def _get_mock_stock_info(self, ticker):
        """
        Generate mock stock information for demo purposes
        
        Parameters:
        -----------
        ticker : str
            Stock symbol
            
        Returns:
        --------
        dict
            Stock information dictionary
        """
        # Mock stock data for demonstration
        stock_data = {
            'AAPL': {
                'shortName': 'Apple Inc.',
                'longName': 'Apple Inc.',
                'sector': 'Technology',
                'industry': 'Consumer Electronics',
                'forwardPE': 28.5,
                'dividendYield': 0.5,
                'beta': 1.2,
                'pegRatio': 1.8
            },
            'MSFT': {
                'shortName': 'Microsoft Corporation',
                'longName': 'Microsoft Corporation',
                'sector': 'Technology',
                'industry': 'Software—Infrastructure',
                'forwardPE': 31.2,
                'dividendYield': 0.8,
                'beta': 0.9,
                'pegRatio': 2.1
            },
            'GOOGL': {
                'shortName': 'Alphabet Inc.',
                'longName': 'Alphabet Inc.',
                'sector': 'Communication Services',
                'industry': 'Internet Content & Information',
                'forwardPE': 25.4,
                'dividendYield': 0.0,
                'beta': 1.1,
                'pegRatio': 1.5
            },
            'AMZN': {
                'shortName': 'Amazon.com, Inc.',
                'longName': 'Amazon.com, Inc.',
                'sector': 'Consumer Cyclical',
                'industry': 'Internet Retail',
                'forwardPE': 47.8,
                'dividendYield': 0.0,
                'beta': 1.3,
                'pegRatio': 2.4
            },
            'TSLA': {
                'shortName': 'Tesla, Inc.',
                'longName': 'Tesla, Inc.',
                'sector': 'Consumer Cyclical',
                'industry': 'Auto Manufacturers',
                'forwardPE': 65.2,
                'dividendYield': 0.0,
                'beta': 2.1,
                'pegRatio': 3.8
            }
        }
        
        # Return stock data if available, otherwise create generic data
        if ticker in stock_data:
            return stock_data[ticker]
        else:
            return {
                'shortName': f'{ticker}',
                'longName': f'{ticker} Inc.',
                'sector': 'Unknown',
                'industry': 'Unknown',
                'forwardPE': np.random.uniform(15, 35),
                'dividendYield': np.random.uniform(0, 3),
                'beta': np.random.uniform(0.8, 1.5),
                'pegRatio': np.random.uniform(1, 3)
            } 