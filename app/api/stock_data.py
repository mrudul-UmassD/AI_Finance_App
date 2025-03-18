import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

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
        # Calculate start and end dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch data from Yahoo Finance
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        # Check if we have data
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Filter relevant columns
        if 'Dividends' in data.columns:
            data = data.drop(['Dividends', 'Stock Splits'], axis=1, errors='ignore')
        
        return data
    
    except Exception as e:
        raise Exception(f"Error fetching stock data for {ticker}: {str(e)}")

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
        # Fetch info from Yahoo Finance
        stock = yf.Ticker(ticker)
        info = stock.info
        
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
        raise Exception(f"Error fetching company info for {ticker}: {str(e)}")

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
        # Get company info
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if sector and 'sector' in info:
            # Get sector
            target_sector = info['sector']
            # This is a placeholder - in a real app we would query a database of stocks by sector
            # For demonstration purposes, return a few well-known stocks in similar sectors
            sectors = {
                'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB'],
                'Financial Services': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
                'Healthcare': ['JNJ', 'PFE', 'MRK', 'UNH', 'ABT'],
                'Consumer Cyclical': ['AMZN', 'HD', 'NKE', 'SBUX', 'MCD'],
                'Communication Services': ['GOOGL', 'FB', 'VZ', 'T', 'NFLX'],
                'Energy': ['XOM', 'CVX', 'BP', 'SHEL', 'COP']
            }
            related = sectors.get(target_sector, [])
        elif not sector and 'industry' in info:
            # Get industry
            target_industry = info['industry']
            # This is a placeholder - in a real app we would query a database of stocks by industry
            industries = {
                'Consumer Electronics': ['AAPL', 'SONY', 'SSNLF', 'HPQ', 'DELL'],
                'Software—Application': ['MSFT', 'ORCL', 'CRM', 'ADBE', 'INTU'],
                'Internet Content & Information': ['GOOGL', 'FB', 'TWTR', 'SNAP', 'PINS'],
                'Banks—Diversified': ['JPM', 'BAC', 'WFC', 'C', 'RY'],
                'Pharmaceutical Retailers': ['CVS', 'WBA', 'RAD', 'RDUS', 'ESALY']
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
        return [] 