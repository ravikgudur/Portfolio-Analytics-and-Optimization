"""
THIS IS A SCRIPT FOR DENOISING CORRELATION MATRIX AND PRODUCING COVARIANCE
for a given set of data

PLEASE NOTE: This script has been created on my personal computer and works 
perfectly fine.

Methodology and Thought Process:
1) For a given data correlation matrix is created and eigen values are estimated
2) these eigen values are fitted to the curve of Marcenko Pastur distribution 
3) For this curve fitting these are the assumptions 
  initial guess values for beta(beta > 0),sigma(sigma >= np.max(eigenvalues))
4) new beta and sigma values are obtained from this curve fitting 
 based on optimization techniques of scipy library.
5) which is used to estimate new denoised correlation 
6) this correlation is convered into covariance based using standard deviations
"""
# Author: Ravi K Gudur

# importing libraries 
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import logging
from functools import wraps
np.set_printoptions(suppress=True, precision=3)


# Configure logging to save logs to a file and also print them to the console
logging.basicConfig(level=logging.INFO,filename='denoise.log',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("denoising_covar.log"),
                              logging.StreamHandler()])

logger = logging.getLogger(__name__)

# Decorator for logging exceptions and warnings
def log_exceptions_warnings(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Exception in {func.__name__}: {e}")
        except Warning as w:
            logger.warning(f"Warning in {func.__name__}: {w}")
    return wrapper

# Function to load data
@log_exceptions_warnings
def load_data(file_path):
    logger.info(f"Loading data from {file_path}")
    data = pd.read_csv(file_path)
    return data

# Function to get mkt cap
@log_exceptions_warnings
def get_mkt_cap(tickers):
    logger.info(f"getting mktcap data ")
    df = pd.DataFrame(columns=['Stock','Marketcap'])
    for ticker in tickers:
        print(ticker)
        try:
            info = yf.Ticker(ticker).fast_info['marketCap']
            df = df._append({'Stock':ticker,'Marketcap':info}, ignore_index=True)
        except:
            print('Error with: ', ticker)
    return df

@log_exceptions_warnings
# Function to retrieve top 30 US stocks by market capitalization
def get_top_100_stocks():

    logger.info(f"fetching historical data from yfinance")
    # Retrieve market data for S&P 500 index
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = sp500['Symbol'].to_list()
    
    # Calculate market capitalization for stocks
    mkt_cap_info = get_mkt_cap(tickers)

    # Sort by market capitalization and select top 30 stocks
    top_100_stocks = mkt_cap_info.sort_values(by=['Marketcap'],ascending = False).head(100)['Stock'].to_list()
        
    return top_100_stocks

# Function to fetch historical stock price data
@log_exceptions_warnings
def fetch_stock_data(symbols, start_date, end_date):
    logger.info(f"fetching historical data from yfinance")
    data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
    return data

# Function to calculate daily returns
@log_exceptions_warnings
def calculate_daily_returns(data):
    logger.info(f"calculating returns from yfinance data")
    returns = data.pct_change().dropna()
    return returns

# Function to handle missing data
@log_exceptions_warnings
def handle_missing_data(data):
    logger.info(" missed data is replaced by column mean")
    mean = data.mean()
    data.fillna(mean,inplace=True)
    return data

# Function to compute array of standard deviations of data
@log_exceptions_warnings
def compute_std_deviations(data):
    logger.info("computing standard deviations of all columns in data")
    std_deviations = data.std().array
    return std_deviations


# Function to compute marcenko pastur pdf
@log_exceptions_warnings
def marcenko_pastur_pdf(x, beta, sigma):
    """
    This function computes pdf based on the formula for marcenko pastur distribution
    
    Parameters:
    - x: Eigenvalues at which to evaluate the PDF.
    - beta: Lower bound of the support.
    - sigma: Upper bound of the support.
    
    Returns:
    - density: The value of the PDF at the given eigenvalues.
    """
    logger.info(" computing marcenko pastur pdf ")
    density = np.sqrt((sigma - x) * (x - beta)) / (2 * np.pi * x * sigma)
    return density

# Objective Function to be used in fitting marcenko pastur distribution
@log_exceptions_warnings   
def objective_function(params, eigenvalues):
    logger.info(" computing objective function  ")
    beta, sigma = params
    expected_eigenvalues = marcenko_pastur_pdf(eigenvalues, beta, sigma)
    # Mean squared error, ignoring NaNs
    error = np.nanmean((eigenvalues - expected_eigenvalues)**2)
    return error

@log_exceptions_warnings
def curve_fit_mp_dist(eigenvalues):
    """
    This function evaluates new paramters( beta, sigma) by fitting the curve of the distribution
    to eigen values

    parameters: eigen values
    Returns: new values of beta and sigma
    """
    logger.info(" computing beta and sigma by fitting distribution curve to eigen values ")
    #initial guess values for beta(beta > 0),sigma(sigma >= np.max(eigenvalues))
    initial_guess = (1 / len(eigenvalues), np.max(eigenvalues))
    result = minimize(objective_function, initial_guess, args=(eigenvalues,),method='Nelder-Mead')
    return result.x
   

@log_exceptions_warnings
def denoising_threshold(new_params):
    """
    This function calulcates threshold using new parameters obtained by curve fitting
    threshold = beta + np.sqrt(sigma/(np.sqrt(sigma)-1)) by formula definition

    beta = new_params[0]
    sigma = new_params[1]

    """
    logger.info(" computing denoising threshold ")
    threshold = new_params[0] + np.sqrt(new_params[1]/(np.sqrt(new_params[1]) - 1))
    return threshold

@log_exceptions_warnings
def corr2covar(corr_mat, std_deviations):
    """
    Estimate the covariance matrix from a given correlation matrix and standard deviations.
    
    Parameters:
    - corr_mat: The correlation matrix.
    - std_deviations: Array of standard deviations of each asset return series.
    
    Returns:
    - covariance_matrix: The estimated covariance matrix.
    """
    logger.info(" computing covariance from correlation")
    diagonal_std = np.diag(std_deviations)
    covar_mat = np.dot(np.dot(diagonal_std, corr_mat), diagonal_std)
    return covar_mat

@log_exceptions_warnings
def denoised_covar_mat(correlation_matrix, threshold,std_deviations):
    """  
    This function returns denoised covariance matrix, given correlation matrix and eigen value threshold
    while keeping trace of original correlation matrix intact

    Parameters: 
    correlation matrix
    threshold - eigen value threshold( integer) below which would be replaced by mean
    std_deviations: Array of standard deviations of each asset return series
    
    """
    logger.info(" computing denoised covariance from correlation for given threshold")

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
    # Denoise eigen values by replacing them with their mean whenever they fall below threshold
    denoised_eigenvalues = np.where(eigenvalues > threshold, eigenvalues, np.mean(eigenvalues))

    # Reconstruct denoised correlation matrix
    denoised_corr_mat = np.dot(np.dot(eigenvectors, np.diag(denoised_eigenvalues)), eigenvectors.T)
    
    # Rescale to preserve trace
    trace_original = np.trace(correlation_matrix)
    trace_denoised = np.trace(denoised_corr_mat)
    denoised_corr_mat *= trace_original / trace_denoised
    
    # convert correlation matrix to covariance matrix
    denoised_covar_mat = corr2covar(denoised_corr_mat, std_deviations)

    return denoised_covar_mat

    
# Main function
def main():
    try:
        # Load data
        # df = load_data(file_path)
        symbols = get_top_100_stocks()  # Stock symbols
        start_date = '2020-01-01'  # Start date of historical data
        end_date = '2021-01-01'  # End date of historical data
        df = fetch_stock_data(symbols, start_date, end_date)
        # Handle missing data
        df = handle_missing_data(df)
        df = calculate_daily_returns(df)
        # calculate correlation matrix
        correlation_matrix = np.corrcoef(df, rowvar=False)
        # ompute the eigenvalues of the correlation matrix
        eigenvalues, _ = np.linalg.eig(correlation_matrix)
        # calculate standard deviations of df
        std_deviations = compute_std_deviations(df)
        # Fit Marcenko-Pastur distribution to the eigenvalues
        new_params = curve_fit_mp_dist(eigenvalues)
        # calculate denoising threshold
        threshold = denoising_threshold(new_params)
        #threshold = new_params[0] + np.sqrt(new_params[1])
        # estimating denoised covariance matrix
        denoised_covariance_matrix = denoised_covar_mat(correlation_matrix, threshold,std_deviations)
        print(denoised_covariance_matrix)
        logger.info("Script execution completed successfully")
    except Exception as e:
        logger.exception(f"Exception occurred during script execution: {e}")

# Execute the main function
if __name__ == "__main__":
    main()
