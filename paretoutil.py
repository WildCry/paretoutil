import yfinance as yf
import pandas as pd
import numpy as np


def log_difference(data):
    """
    Calculate the log difference of a pandas Series, DataFrame, or NumPy array.

    Parameters:
    - data: pd.Series, pd.DataFrame, or np.ndarray, the data for which to calculate the log difference.

    Returns:
    - The log differences of the input data, with the same type as the input.
    """
    # Determine the type of the input data
    if isinstance(data, pd.Series):
        # Use np.log to calculate the natural log of the series
        log_data = np.log(data)
        # Calculate the difference between each log value and its predecessor
        result = log_data.diff()

    elif isinstance(data, pd.DataFrame):
        # Apply the log difference calculation to each column in the DataFrame
        log_data = np.log(data)
        result = log_data.diff()

    elif isinstance(data, np.ndarray):
        # Calculate the log of the NumPy array
        log_data = np.log(data)
        # Calculate the difference between each log value and its predecessor along the first axis
        result = np.diff(log_data, axis=0)
        # To align with the input's shape, we need to add a row of NaNs or handle the shape discrepancy
        result = np.insert(result, 0, np.nan, axis=0)

    else:
        raise ValueError(
            "Input must be a pd.Series, pd.DataFrame, or np.ndarray.")

    return result


def fetch_series(ticker_obj: yf.Ticker, period: str) -> pd.Series:
    # Generate a unique cache key based on ticker symbol and period
    cache_key = f"{ticker_obj.ticker}_{period}_closing"

    # Attempt to retrieve cached series
    series = cache_manager.get_series(cache_key)

    if series is None:
        # Fetch new data if not found in cache
        data = ticker_obj.history(period=period)['Close']

        # Update cache
        cache_manager.set_series(cache_key, data)

        return data
    else:
        # Return cached series if available
        return series


class CacheManager:
    '''
    Will cache api reqiests for time series.
    to reinitialize cache run 

    cache_manager = CacheManager()
    '''

    def __init__(self):
        self.cache = {}

    def get_series(self, key: str) -> pd.Series:
        """Retrieve series from cache if available."""
        return self.cache.get(key, None)

    def set_series(self, key: str, series: pd.Series):
        """Store series in cache."""
        self.cache[key] = series


class MarketPortfolio(yf.Ticker):
    '''
    Defines what will be used as the market portfolio. standard is OSEBX.OL
    '''

    def __init__(self, ticker='OSEBX.OL'):
        super().__init__(ticker)

    def series(self, period: str = '1y'):
        return fetch_series(ticker_obj=self, period=period)


# Initialize market portfolio
market_portfolio = MarketPortfolio('OSEBX.OL')

# Initialize a global cache manager
cache_manager = CacheManager()


class Stock(yf.Ticker):

    def __init__(self, ticker, session=None):
        super().__init__(ticker, session)

    def ClosingPrices(self, period: str = '1y') -> pd.Series:
        return fetch_series(self, period=period)

    def Beta(self, index_symbol: str = 'OSEBX.OL', period: str = '1y') -> float:
        # Fetch closing prices and calculate log differences for the company
        company_prices = self.ClosingPrices(period=period)
        company_returns = log_difference(company_prices)

        # Fetch closing prices and calculate log differences for the market index
        index_prices = market_portfolio.series(period=period)
        index_returns = log_difference(index_prices)

        # Drop any NaN values that might have been introduced by the log_difference calculation
        company_returns = company_returns.dropna()
        index_returns = index_returns.dropna()

        # Align both series to the same dates
        aligned_company_returns, aligned_index_returns = company_returns.align(
            index_returns, join='inner')

        # Calculate covariance between company and index returns
        covariance = np.cov(aligned_company_returns,
                            aligned_index_returns)[0][1]

        # Calculate variance of the market index returns
        variance = np.var(aligned_index_returns)

        # Calculate beta
        beta = covariance / variance

        return beta

    def CAPM(self, rf: float = 0.03787) -> float:
        rm = market_portfolio.series(period='max')
        rm = rm.resample('y').last().pct_change().mean()
        return rf + self.Beta(period='max') * (rm - rf)

    def interest_on_debt(self) -> pd.Series:
        interest_exp = self.financials.loc['Interest Expense']
        debt = self.balance_sheet.loc['Total Debt']

        return interest_exp / debt

    def WACC(self):
        Market_cap = self.basic_info['marketCap']
        debt = self.balance_sheet.loc['Total Debt'].iloc[0]
        debt_rate = debt/(Market_cap + debt)

        return (1-debt_rate)*self.CAPM() + debt_rate*(1-0.22)*self.interest_on_debt().iloc[0]

    def EV(self) -> pd.Series:
        '''
        Enterprise value

        `Market Capitalization + Total Debt - Cash and Cash equivalents`

        returns:
        Pandas series for with yearly data.
        '''
        Market_cap = self.basic_info['marketCap']
        total_debt = self.balance_sheet.loc['Current Debt']
        cash_and_cash_equivalents = self.balance_sheet.loc['Cash And Cash Equivalents']

        return Market_cap + total_debt - cash_and_cash_equivalents
