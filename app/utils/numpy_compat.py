"""
Compatibility module for numpy functions to ensure compatibility with different Python versions
"""

def is_nan(value):
    """
    Check if a value is NaN in a way that's compatible with all numpy versions
    
    Parameters:
    -----------
    value : any
        Value to check
        
    Returns:
    --------
    bool
        True if value is NaN, False otherwise
    """
    try:
        # Try to compare with itself - NaN is the only value that doesn't equal itself
        return value != value
    except:
        # If comparison fails, it's not a NaN
        return False

def get_nan():
    """
    Get a NaN value in a way that's compatible with all numpy versions
    
    Returns:
    --------
    float
        NaN value
    """
    try:
        import numpy as np
        return np.nan
    except (ImportError, AttributeError):
        # Fallback to create NaN without numpy
        return float('nan')

def safe_divide(a, b):
    """
    Safe division that returns NaN on divide by zero
    
    Parameters:
    -----------
    a : number
        Numerator
    b : number
        Denominator
        
    Returns:
    --------
    float
        Result of division or NaN if b is zero
    """
    try:
        if b == 0:
            return get_nan()
        return a / b
    except:
        return get_nan() 