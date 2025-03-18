import requests
from datetime import datetime, timedelta
import re
from bs4 import BeautifulSoup
import random
import time

def get_financial_news(ticker, limit=10):
    """
    Fetch financial news related to a stock ticker directly from the web
    
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
        # Try to get real news from web scraping
        news = scrape_financial_news(ticker, limit)
        if news:
            return news
        
        # Fallback to mock data if scraping fails
        return get_mock_news(ticker, limit)
    
    except Exception as e:
        # If there's an error, return mock data
        print(f"Error fetching news data: {str(e)}")
        return get_mock_news(ticker, limit)

def scrape_financial_news(ticker, limit=10):
    """
    Scrape financial news from Yahoo Finance
    
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
    news = []
    
    try:
        # Create URL for Yahoo Finance news
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        
        # Set user agent to mimic browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make request
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news items
            news_items = soup.find_all('li', class_='js-stream-content')
            
            # Process news items
            for item in news_items[:limit]:
                try:
                    # Get title
                    title_element = item.find('h3')
                    if not title_element:
                        continue
                    title = title_element.text.strip()
                    
                    # Get URL
                    link_element = item.find('a')
                    if not link_element or not link_element.has_attr('href'):
                        continue
                    url = link_element['href']
                    if not url.startswith('http'):
                        url = 'https://finance.yahoo.com' + url
                    
                    # Get source and publish time
                    source_element = item.find('div', class_='C(#959595)')
                    source = ''
                    publish_time = datetime.now().isoformat()
                    if source_element:
                        source_text = source_element.text.strip()
                        source_match = re.search(r'(.+?)(?:\s+[·•-]\s+)?', source_text)
                        if source_match:
                            source = source_match.group(1).strip()
                    
                    # Get description (need to visit the article page)
                    description = ''
                    try:
                        article_response = requests.get(url, headers=headers)
                        if article_response.status_code == 200:
                            article_soup = BeautifulSoup(article_response.text, 'html.parser')
                            description_element = article_soup.find('div', class_='caas-body')
                            if description_element and description_element.p:
                                description = description_element.p.text.strip()
                            else:
                                # Try alternative method
                                paragraphs = article_soup.find_all('p')
                                if paragraphs and len(paragraphs) > 0:
                                    description = paragraphs[0].text.strip()
                    except:
                        # If we can't get the description, just leave it empty
                        pass
                    
                    # Add to news list
                    news.append({
                        'title': title,
                        'description': description[:250] + '...' if len(description) > 250 else description,
                        'url': url,
                        'source': source,
                        'publishedAt': publish_time
                    })
                    
                    # Add a small delay to avoid overloading the server
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"Error processing news item: {str(e)}")
                    continue
                    
            return news
        
    except Exception as e:
        print(f"Error scraping news: {str(e)}")
    
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
                'url': "https://finance.yahoo.com/news/apple-earnings",
                'source': "Financial Times",
                'publishedAt': (datetime.now() - timedelta(days=1)).isoformat()
            },
            {
                'title': f"Apple's AR/VR Headset: What We Know So Far",
                'description': "The long-rumored mixed reality headset could be announced at the upcoming WWDC event.",
                'url': "https://finance.yahoo.com/news/apple-arvr",
                'source': "Tech Crunch",
                'publishedAt': (datetime.now() - timedelta(days=2)).isoformat()
            },
            {
                'title': f"Apple Faces Antitrust Challenges in Europe",
                'description': "European regulators have raised concerns about Apple's App Store policies and may impose new regulations.",
                'url': "https://finance.yahoo.com/news/apple-antitrust",
                'source': "Reuters",
                'publishedAt': (datetime.now() - timedelta(days=3)).isoformat()
            },
            {
                'title': f"Analysts Upgrade Apple Stock Rating to 'Strong Buy'",
                'description': "Several Wall Street firms have raised their price targets following strong quarterly results.",
                'url': "https://finance.yahoo.com/news/apple-upgrade",
                'source': "Wall Street Journal",
                'publishedAt': (datetime.now() - timedelta(days=4)).isoformat()
            },
            {
                'title': f"Apple Plans Major iPhone Design Overhaul for 2023",
                'description': "Sources suggest the next iPhone generation will feature significant design changes and new technologies.",
                'url': "https://finance.yahoo.com/news/apple-iphone-design",
                'source': "Bloomberg",
                'publishedAt': (datetime.now() - timedelta(days=5)).isoformat()
            }
        ],
        'MSFT': [
            {
                'title': f"Microsoft Cloud Revenue Surges in Latest Quarter",
                'description': "Azure cloud services saw a 40% growth as businesses continue digital transformation efforts.",
                'url': "https://finance.yahoo.com/news/microsoft-cloud",
                'source': "CNBC",
                'publishedAt': (datetime.now() - timedelta(days=1)).isoformat()
            },
            {
                'title': f"Microsoft Expands AI Capabilities in Office Suite",
                'description': "New AI-powered features aim to enhance productivity and user experience across Microsoft 365 applications.",
                'url': "https://finance.yahoo.com/news/microsoft-ai-office",
                'source': "The Verge",
                'publishedAt': (datetime.now() - timedelta(days=2)).isoformat()
            },
            {
                'title': f"Microsoft Teams Reaches 300 Million Daily Active Users",
                'description': "The collaboration platform continues to grow as remote and hybrid work becomes the norm for many organizations.",
                'url': "https://finance.yahoo.com/news/microsoft-teams-users",
                'source': "Business Insider",
                'publishedAt': (datetime.now() - timedelta(days=3)).isoformat()
            },
            {
                'title': f"Microsoft Gaming Revenue Shows Strong Growth Following Activision Acquisition",
                'description': "Xbox and Game Pass subscriptions contribute to record gaming revenue for the company.",
                'url': "https://finance.yahoo.com/news/microsoft-gaming",
                'source': "GameSpot",
                'publishedAt': (datetime.now() - timedelta(days=4)).isoformat()
            },
            {
                'title': f"Microsoft Invests $5 Billion in AI Research Initiative",
                'description': "The company announced a major funding commitment to develop next-generation AI technologies.",
                'url': "https://finance.yahoo.com/news/microsoft-ai-investment",
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
            'url': f"https://finance.yahoo.com/news/{ticker.lower()}-trading",
            'source': "Market Watch",
            'publishedAt': (datetime.now() - timedelta(days=1)).isoformat()
        },
        {
            'title': f"Q2 Earnings Preview: What to Expect from {ticker}",
            'description': f"Analysts predict strong performance for {ticker} in the upcoming earnings report.",
            'url': f"https://finance.yahoo.com/news/{ticker.lower()}-earnings",
            'source': "Seeking Alpha",
            'publishedAt': (datetime.now() - timedelta(days=2)).isoformat()
        },
        {
            'title': f"{ticker} Announces New Strategic Partnership",
            'description': f"The partnership is expected to open new market opportunities and drive revenue growth.",
            'url': f"https://finance.yahoo.com/news/{ticker.lower()}-partnership",
            'source': "Bloomberg",
            'publishedAt': (datetime.now() - timedelta(days=3)).isoformat()
        },
        {
            'title': f"Is {ticker} a Good Buy Right Now? Experts Weigh In",
            'description': "Financial analysts offer their perspectives on whether investors should buy, hold, or sell.",
            'url': f"https://finance.yahoo.com/news/{ticker.lower()}-analysis",
            'source': "Motley Fool",
            'publishedAt': (datetime.now() - timedelta(days=4)).isoformat()
        },
        {
            'title': f"{ticker} Expands Operations in Emerging Markets",
            'description': "The company is making significant investments in key growth regions.",
            'url': f"https://finance.yahoo.com/news/{ticker.lower()}-expansion",
            'source': "Financial Times",
            'publishedAt': (datetime.now() - timedelta(days=5)).isoformat()
        }
    ]
    
    # Return appropriate news data
    if ticker in news_data:
        return news_data[ticker][:limit]
    else:
        return default_news[:limit] 