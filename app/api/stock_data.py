import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import time
import random

def get_stock_data(ticker, days):
    """
    Fetch historical stock data for a given ticker
    
    Parameters:
    -----------
    ticker : str
        Stock symbol (e.g., 'AAPL', 'MSFT')
    days : int
        Number of days of historical data to fetch
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the historical stock data
    """
    try:
        # Try to get data from Yahoo Finance API
        data = get_data_from_yfinance(ticker, days)
        
        if data is not None and not data.empty:
            return data
        
        # If YFinance fails, try web scraping
        print(f"YFinance API failed for {ticker}, trying web scraping...")
        data = scrape_stock_data(ticker, days)
        
        if data is not None and not data.empty:
            return data
            
        # If both methods fail, raise exception
        raise ValueError(f"No data found for ticker {ticker}")
    
    except Exception as e:
        raise Exception(f"Error fetching stock data for {ticker}: {str(e)}")

def get_data_from_yfinance(ticker, days):
    """
    Fetch stock data from Yahoo Finance API
    """
    try:
        # Calculate start and end dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch data from Yahoo Finance
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        # Check if we have data
        if data.empty:
            return None
        
        # Filter relevant columns
        if 'Dividends' in data.columns:
            data = data.drop(['Dividends', 'Stock Splits'], axis=1, errors='ignore')
        
        return data
        
    except Exception as e:
        print(f"YFinance error: {str(e)}")
        return None

def scrape_stock_data(ticker, days):
    """
    Scrape stock data from Yahoo Finance website as fallback
    """
    try:
        # Calculate period
        if days <= 30:
            period = '1mo'
        elif days <= 90:
            period = '3mo'
        elif days <= 180:
            period = '6mo'
        elif days <= 365:
            period = '1y'
        elif days <= 730:
            period = '2y'
        elif days <= 1825:
            period = '5y'
        else:
            period = 'max'
        
        # Create URL for historical data
        url = f"https://finance.yahoo.com/quote/{ticker}/history?period1={int((datetime.now() - timedelta(days=days)).timestamp())}&period2={int(datetime.now().timestamp())}&interval=1d"
        
        # Set user agent to mimic browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make request
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            return None
            
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the data table
        table = soup.find('table', {'data-test': 'historical-prices'})
        if not table:
            return None
            
        # Extract data
        rows = table.find_all('tr')
        if not rows or len(rows) <= 1:  # Skip header
            return None
            
        # Create lists for data
        dates = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        # Parse rows
        for row in rows[1:]:  # Skip header row
            columns = row.find_all('td')
            if len(columns) < 6:
                continue
                
            # Check if it's a dividend or split row
            if 'Dividend' in columns[1].text or 'Split' in columns[1].text:
                continue
                
            # Extract date
            date_text = columns[0].text.strip()
            try:
                date = pd.to_datetime(date_text)
            except:
                continue
                
            # Extract price data, replacing any non-numeric with NaN
            def safe_parse(val):
                try:
                    return float(val.replace(',', ''))
                except:
                    return np.nan
            
            try:
                open_price = safe_parse(columns[1].text.strip())
                high_price = safe_parse(columns[2].text.strip())
                low_price = safe_parse(columns[3].text.strip())
                close_price = safe_parse(columns[4].text.strip())
                
                # Volume might have K, M, B suffixes
                volume_text = columns[5].text.strip()
                volume = convert_volume(volume_text)
            except:
                continue
                
            # Add data to lists
            dates.append(date)
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            volumes.append(volume)
            
        # Create DataFrame
        data = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': volumes
        }, index=dates)
        
        # Sort by date (most recent first, like yfinance)
        data = data.sort_index(ascending=False)
        
        return data
        
    except Exception as e:
        print(f"Web scraping error: {str(e)}")
        return None

def convert_volume(volume_text):
    """
    Convert volume text (e.g., '1.5M', '45K') to integer
    """
    volume_text = volume_text.replace(',', '')
    
    if 'K' in volume_text:
        return float(volume_text.replace('K', '')) * 1000
    elif 'M' in volume_text:
        return float(volume_text.replace('M', '')) * 1000000
    elif 'B' in volume_text:
        return float(volume_text.replace('B', '')) * 1000000000
    else:
        try:
            return float(volume_text)
        except:
            return 0

def get_stock_info(ticker):
    """
    Fetch company information for a given ticker
    
    Parameters:
    -----------
    ticker : str
        Stock symbol (e.g., 'AAPL', 'MSFT')
    
    Returns:
    --------
    dict
        Dictionary containing company information
    """
    try:
        # Try to get info from Yahoo Finance API
        info = get_info_from_yfinance(ticker)
        if info:
            return info
            
        # If YFinance fails, try web scraping
        print(f"YFinance API failed for {ticker} info, trying web scraping...")
        info = scrape_stock_info(ticker)
        if info:
            return info
            
        # If both methods fail, return mock data
        return get_mock_stock_info(ticker)
    
    except Exception as e:
        print(f"Error fetching company info for {ticker}: {str(e)}")
        return get_mock_stock_info(ticker)

def get_info_from_yfinance(ticker):
    """
    Fetch company info from Yahoo Finance API
    """
    try:
        # Fetch info from Yahoo Finance
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Check if we have info
        if not info or len(info) == 0:
            return None
        
        # Extract relevant fields
        relevant_info = {
            'shortName': info.get('shortName', ''),
            'longName': info.get('longName', ''),
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
            'website': info.get('website', ''),
            'marketCap': info.get('marketCap', None),
            'forwardPE': info.get('forwardPE', None),
            'dividendYield': info.get('dividendYield', None) * 100 if info.get('dividendYield') else None,
            'beta': info.get('beta', None),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', None),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', None),
            'averageVolume': info.get('averageVolume', None),
            'trailingEps': info.get('trailingEps', None),
            'pegRatio': info.get('pegRatio', None),
            'shortRatio': info.get('shortRatio', None)
        }
        
        return relevant_info
        
    except Exception as e:
        print(f"YFinance info error: {str(e)}")
        return None

def scrape_stock_info(ticker):
    """
    Scrape company info from Yahoo Finance website as fallback
    """
    try:
        # Create URL for quote page
        url = f"https://finance.yahoo.com/quote/{ticker}"
        
        # Set user agent to mimic browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make request
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            return None
            
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Init info dictionary
        info = {
            'shortName': '',
            'longName': '',
            'sector': '',
            'industry': '',
            'website': '',
            'marketCap': None,
            'forwardPE': None,
            'dividendYield': None,
            'beta': None,
            'fiftyTwoWeekHigh': None,
            'fiftyTwoWeekLow': None,
            'averageVolume': None,
            'trailingEps': None,
            'pegRatio': None,
            'shortRatio': None
        }
        
        # Get company name
        name_element = soup.find('h1')
        if name_element:
            info['shortName'] = name_element.text.strip()
            info['longName'] = info['shortName']
        
        # Get overview data
        overview_div = soup.find('div', id='quote-summary')
        if overview_div:
            rows = overview_div.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) == 2:
                    label = cols[0].text.strip()
                    value = cols[1].text.strip()
                    
                    if label == 'Market Cap':
                        info['marketCap'] = parse_numeric_value(value)
                    elif label == 'Beta (5Y Monthly)':
                        info['beta'] = parse_numeric_value(value)
                    elif label == 'PE Ratio (TTM)':
                        info['forwardPE'] = parse_numeric_value(value)
                    elif label == '52 Week Range':
                        try:
                            parts = value.split('-')
                            if len(parts) == 2:
                                info['fiftyTwoWeekLow'] = parse_numeric_value(parts[0])
                                info['fiftyTwoWeekHigh'] = parse_numeric_value(parts[1])
                        except:
                            pass
                    elif label == 'Avg. Volume':
                        info['averageVolume'] = parse_numeric_value(value)
                    elif label == 'Forward Dividend & Yield':
                        try:
                            match = re.search(r'\((.*?)%\)', value)
                            if match:
                                info['dividendYield'] = float(match.group(1))
                        except:
                            pass
        
        return info
        
    except Exception as e:
        print(f"Web scraping info error: {str(e)}")
        return None

def parse_numeric_value(text):
    """
    Parse numeric values from text with possible suffixes (K, M, B, T)
    """
    try:
        text = text.replace(',', '')
        
        # Handle suffixes
        if 'K' in text:
            return float(text.replace('K', '')) * 1000
        elif 'M' in text:
            return float(text.replace('M', '')) * 1000000
        elif 'B' in text:
            return float(text.replace('B', '')) * 1000000000
        elif 'T' in text:
            return float(text.replace('T', '')) * 1000000000000
        else:
            return float(text)
    except:
        return None

def get_mock_stock_info(ticker):
    """
    Generate mock stock information for demonstration purposes
    
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

def get_related_stocks(ticker, sector=True):
    """
    Find related stocks based on sector or industry
    
    Parameters:
    -----------
    ticker : str
        Stock symbol (e.g., 'AAPL', 'MSFT')
    sector : bool
        If True, find stocks in the same sector; otherwise find stocks in the same industry
    
    Returns:
    --------
    list
        List of related stock symbols
    """
    try:
        # First, get company info to determine sector/industry
        stock_info = get_stock_info(ticker)
        
        if not stock_info:
            return []
            
        if sector and 'sector' in stock_info and stock_info['sector']:
            # Get sector
            target_sector = stock_info['sector']
            # This is a placeholder - in a real app we would query more data
            # For demonstration purposes, return a few well-known stocks in similar sectors
            sectors = {
                'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB', 'NVDA', 'INTC'],
                'Financial Services': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'V', 'MA'],
                'Healthcare': ['JNJ', 'PFE', 'MRK', 'UNH', 'ABT', 'TMO', 'LLY'],
                'Consumer Cyclical': ['AMZN', 'HD', 'NKE', 'SBUX', 'MCD', 'TGT', 'LULU'],
                'Communication Services': ['GOOGL', 'META', 'VZ', 'T', 'NFLX', 'DIS', 'TMUS'],
                'Energy': ['XOM', 'CVX', 'BP', 'SHEL', 'COP', 'SLB', 'EOG']
            }
            related = sectors.get(target_sector, [])
        elif not sector and 'industry' in stock_info and stock_info['industry']:
            # Get industry
            target_industry = stock_info['industry']
            # This is a placeholder - in a real app we would query more data
            industries = {
                'Consumer Electronics': ['AAPL', 'SONY', 'SSNLF', 'HPQ', 'DELL', 'FIT', 'HEAR'],
                'Software—Application': ['MSFT', 'ORCL', 'CRM', 'ADBE', 'INTU', 'NOW', 'WDAY'],
                'Internet Content & Information': ['GOOGL', 'META', 'TWTR', 'SNAP', 'PINS', 'MTCH', 'IAC'],
                'Banks—Diversified': ['JPM', 'BAC', 'WFC', 'C', 'RY', 'TD', 'BCS'],
                'Pharmaceutical Retailers': ['CVS', 'WBA', 'RAD', 'RDUS', 'ESALY', 'RBGLY', 'PBH']
            }
            related = industries.get(target_industry, [])
        else:
            related = []
        
        # Remove the original ticker from the list
        if ticker in related:
            related.remove(ticker)
            
        return related
    
    except Exception as e:
        # Return empty list if there's an error
        print(f"Error finding related stocks: {str(e)}")
        return [] 