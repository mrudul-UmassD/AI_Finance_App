import requests
from datetime import datetime, timedelta
import os
import json

# Placeholder for NewsAPI key - in production, store this in environment variables
NEWS_API_KEY = "YOUR_NEWS_API_KEY"  # Replace with actual API key

def get_financial_news(ticker, limit=10):
    """
    Fetch financial news related to a stock ticker
    
    Parameters:
    -----------
    ticker : str
        Stock symbol (e.g., 'AAPL', 'MSFT')
    limit : int
        Maximum number of news articles to return
        
    Returns:
    --------
    list
        List of news articles as dictionaries
    """
    try:
        # In a real application, use the actual API
        if NEWS_API_KEY != "YOUR_NEWS_API_KEY":
            return get_news_from_api(ticker, limit)
        else:
            # Return mock data for demo purposes
            return get_mock_news(ticker, limit)
    
    except Exception as e:
        # If there's an error, return empty list
        print(f"Error fetching news data: {str(e)}")
        return []

def get_news_from_api(ticker, limit=10):
    """
    Fetch financial news from the NewsAPI
    
    Parameters:
    -----------
    ticker : str
        Stock symbol (e.g., 'AAPL', 'MSFT')
    limit : int
        Maximum number of news articles to return
        
    Returns:
    --------
    list
        List of news articles as dictionaries
    """
    # Calculate date range (last 7 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Format dates for API
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')
    
    # Prepare company name for query (e.g., 'AAPL' -> 'Apple', 'MSFT' -> 'Microsoft')
    companies = {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Google',
        'AMZN': 'Amazon',
        'FB': 'Facebook',
        'TSLA': 'Tesla',
        'NFLX': 'Netflix',
        'NVDA': 'NVIDIA'
    }
    
    # Use ticker symbol as query if not in our list
    query = companies.get(ticker, ticker)
    
    # Construct URL
    url = f"https://newsapi.org/v2/everything"
    params = {
        'q': f"{query} stock",
        'from': from_date,
        'to': to_date,
        'sortBy': 'publishedAt',
        'language': 'en',
        'apiKey': NEWS_API_KEY
    }
    
    # Make request
    response = requests.get(url, params=params)
    data = response.json()
    
    # Check if we have results
    if response.status_code == 200 and data.get('articles'):
        articles = data['articles'][:limit]
        
        # Format the results
        results = []
        for article in articles:
            results.append({
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'url': article.get('url', ''),
                'source': article.get('source', {}).get('name', ''),
                'publishedAt': article.get('publishedAt', '')
            })
        
        return results
    else:
        return []

def get_mock_news(ticker, limit=5):
    """
    Generate mock news data for demonstration purposes
    
    Parameters:
    -----------
    ticker : str
        Stock symbol (e.g., 'AAPL', 'MSFT')
    limit : int
        Maximum number of news articles to return
        
    Returns:
    --------
    list
        List of news articles as dictionaries
    """
    # Mock news data keyed by ticker
    news_data = {
        'AAPL': [
            {
                'title': f"Apple Reports Record Q2 Earnings, Beats Expectations",
                'description': "The tech giant reported a 12% increase in revenue and announced a new $100 billion stock buyback program.",
                'url': "https://example.com/news/1",
                'source': "Financial Times",
                'publishedAt': (datetime.now() - timedelta(days=1)).isoformat()
            },
            {
                'title': f"Apple's AR/VR Headset: What We Know So Far",
                'description': "The long-rumored mixed reality headset could be announced at the upcoming WWDC event.",
                'url': "https://example.com/news/2",
                'source': "Tech Crunch",
                'publishedAt': (datetime.now() - timedelta(days=2)).isoformat()
            },
            {
                'title': f"Apple Faces Antitrust Challenges in Europe",
                'description': "European regulators have raised concerns about Apple's App Store policies and may impose new regulations.",
                'url': "https://example.com/news/3",
                'source': "Reuters",
                'publishedAt': (datetime.now() - timedelta(days=3)).isoformat()
            },
            {
                'title': f"Analysts Upgrade Apple Stock Rating to 'Strong Buy'",
                'description': "Several Wall Street firms have raised their price targets following strong quarterly results.",
                'url': "https://example.com/news/4",
                'source': "Wall Street Journal",
                'publishedAt': (datetime.now() - timedelta(days=4)).isoformat()
            },
            {
                'title': f"Apple Plans Major iPhone Design Overhaul for 2023",
                'description': "Sources suggest the next iPhone generation will feature significant design changes and new technologies.",
                'url': "https://example.com/news/5",
                'source': "Bloomberg",
                'publishedAt': (datetime.now() - timedelta(days=5)).isoformat()
            }
        ],
        'MSFT': [
            {
                'title': f"Microsoft Cloud Revenue Surges in Latest Quarter",
                'description': "Azure cloud services saw a 40% growth as businesses continue digital transformation efforts.",
                'url': "https://example.com/news/6",
                'source': "CNBC",
                'publishedAt': (datetime.now() - timedelta(days=1)).isoformat()
            },
            {
                'title': f"Microsoft Expands AI Capabilities in Office Suite",
                'description': "New AI-powered features aim to enhance productivity and user experience across Microsoft 365 applications.",
                'url': "https://example.com/news/7",
                'source': "The Verge",
                'publishedAt': (datetime.now() - timedelta(days=2)).isoformat()
            },
            {
                'title': f"Microsoft Teams Reaches 300 Million Daily Active Users",
                'description': "The collaboration platform continues to grow as remote and hybrid work becomes the norm for many organizations.",
                'url': "https://example.com/news/8",
                'source': "Business Insider",
                'publishedAt': (datetime.now() - timedelta(days=3)).isoformat()
            },
            {
                'title': f"Microsoft Gaming Revenue Shows Strong Growth Following Activision Acquisition",
                'description': "Xbox and Game Pass subscriptions contribute to record gaming revenue for the company.",
                'url': "https://example.com/news/9",
                'source': "GameSpot",
                'publishedAt': (datetime.now() - timedelta(days=4)).isoformat()
            },
            {
                'title': f"Microsoft Invests $5 Billion in AI Research Initiative",
                'description': "The company announced a major funding commitment to develop next-generation AI technologies.",
                'url': "https://example.com/news/10",
                'source': "MIT Technology Review",
                'publishedAt': (datetime.now() - timedelta(days=5)).isoformat()
            }
        ]
    }
    
    # Default news for any ticker not in our mock data
    default_news = [
        {
            'title': f"{ticker} Stock Sees Unusual Trading Activity",
            'description': f"Trading volume for {ticker} was significantly higher than normal, prompting analyst speculation.",
            'url': "https://example.com/news/d1",
            'source': "Market Watch",
            'publishedAt': (datetime.now() - timedelta(days=1)).isoformat()
        },
        {
            'title': f"Q2 Earnings Preview: What to Expect from {ticker}",
            'description': f"Analysts predict strong performance for {ticker} in the upcoming earnings report.",
            'url': "https://example.com/news/d2",
            'source': "Seeking Alpha",
            'publishedAt': (datetime.now() - timedelta(days=2)).isoformat()
        },
        {
            'title': f"{ticker} Announces New Strategic Partnership",
            'description': f"The partnership is expected to open new market opportunities and drive revenue growth.",
            'url': "https://example.com/news/d3",
            'source': "Bloomberg",
            'publishedAt': (datetime.now() - timedelta(days=3)).isoformat()
        },
        {
            'title': f"Is {ticker} a Good Buy Right Now? Experts Weigh In",
            'description': "Financial analysts offer their perspectives on whether investors should buy, hold, or sell.",
            'url': "https://example.com/news/d4",
            'source': "Motley Fool",
            'publishedAt': (datetime.now() - timedelta(days=4)).isoformat()
        },
        {
            'title': f"{ticker} Expands Operations in Emerging Markets",
            'description': "The company is making significant investments in key growth regions.",
            'url': "https://example.com/news/d5",
            'source': "Financial Times",
            'publishedAt': (datetime.now() - timedelta(days=5)).isoformat()
        }
    ]
    
    # Return appropriate news data
    if ticker in news_data:
        return news_data[ticker][:limit]
    else:
        return default_news[:limit] 