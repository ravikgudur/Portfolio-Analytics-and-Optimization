# This file has all four methods for Efficient Frontier and Portfolio Optimization
# 1) Monte Carlo
# 2) Black Litterman
# 3) CLT
# 4) Hierarchical Parity

# Author: Ravi k Gudur

# Optimization methods for  Efficient Frontier

# Import libraries 

import yfinance as yf
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from functools import wraps
import logging
import warnings
warnings.filterwarnings("error")



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


# Fuction for Montecarlo Method for Efficient Frontier
@log_exceptions_warnings
def montecarlo_port_opt(returns):
    logger.info("Running Monte Carlo Method for Efficient Frontier")
    # expected returns
    mean_daily_returns = returns.mean()
    mu = (1 + mean_daily_returns) ** 252 - 1
    mu = mu.values
    expected_returns=mu

    # covariance
    covar = returns.cov()
    sigma = covar*252
    sigma = sigma.values
    covariance_matrix=sigma

    # Number of assets
    n_assets = len(expected_returns)

    # Number of portfolios to simulate
    n_portfolios = 10000

    # Generate random portfolios
    portfolio_weights = np.zeros((n_portfolios,n_assets))
    portfolio_returns = np.zeros(n_portfolios)
    portfolio_stds = np.zeros(n_portfolios)
    sharpe_ratios = np.zeros(n_portfolios)

    for ind in range(n_portfolios):
        # Generate random portfolio weights
        weights = np.array(np.random.rand(n_assets))
        weights /= np.sum(weights)

        # Calculate portfolio return, standard deviation,sharpe ratio
        portfolio_returns[ind] = np.sum(expected_returns * weights)
        portfolio_stds[ind] = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights.T)))
        sharpe_ratios[ind] = np.nan if portfolio_stds[ind]==0  else portfolio_returns[ind]/sharpe_ratios[ind]
        # Save portfolio weights
        portfolio_weights[ind] = weights

    # Filter portfolios that satisfy the weight constraints
    valid_portfolios = (portfolio_weights.sum(axis=1) == 1) & (portfolio_weights >= 0).all(axis=1)
    portfolio_weights = portfolio_weights[valid_portfolios]
    portfolio_returns = portfolio_returns[valid_portfolios]
    portfolio_stds = portfolio_stds[valid_portfolios]
    sharpe_ratios = sharpe_ratios[valid_portfolios]

    max_sharpe = sharpe_ratios.max()
    max_sharpe_weights = portfolio_weights[sharpe_ratios.argmax()]
    max_sharpe_std = portfolio_stds[sharpe_ratios.argmax()]
    max_sharpe_return = portfolio_returns[sharpe_ratios.argmax()]

    min_variance_std = portfolio_stds.min()
    min_variance_return = portfolio_returns[portfolio_stds.argmin()]
    min_var_weights = portfolio_weights[portfolio_stds.argmin()]

    # Plot the efficient frontier
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_stds, portfolio_returns, marker='o', color='blue', label='Monte Carlo Efficient Frontier')
    plt.scatter(min_variance_std, min_variance_return, marker='o', color='red', label='Minimum Variance Portfolio')
    plt.scatter(max_sharpe_std, max_sharpe_return, marker='o', color='green', label='Maximum Sharpe Ratio Portfolio')
    plt.plot(portfolio_stds, portfolio_returns, color='magenta', linestyle='--', label='Efficient Frontier Line')
    plt.xlabel('Portfolio Standard Deviation')
    plt.ylabel('Portfolio Return')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return portfolio_returns, portfolio_stds, min_variance_return, min_variance_std
  

# Fuction for Black Litterman Method for Efficient Frontier
@log_exceptions_warnings
def black_litterman_opt(returns,tau=0.5):
    """
    returns : Pandas Data Frame
    mu: ndarray -> expected returns
    sigma:  ndarray -> covariance_matrix
    pi : ndarray-> prior_returns
    omega: ndarray - Confidence levels of prior uncertainty in the form of covariance matrix
    tau: float - Risk aversion parameter

    """
    logger.info(f"Running Black Litterman Method for Efficient Frontier")

    # expected returns
    mean_daily_returns = returns.mean()
    mu = (1 + mean_daily_returns) ** 252 - 1
    mu = mu.values

    # covariance
    covar = returns.cov()
    sigma = covar*252
    sigma = sigma.values

    # Number of assets
    n_assets = len(mu)

    # investor's views, confidence levels and prior beliefs are created randomly
    pi = np.array(np.random.rand(n_assets))
    omega =  np.diag(np.array(np.random.rand(n_assets)))

    # portfolio weights defined as variable for number of assets
    weights = cp.Variable(n_assets)

    # Constraint_1: All weights are positive
    constraints = [weights >= 0]

    # Constraint_2: sum of weights is one
    constraints.append(cp.sum(weights) == 1)

    # Modified covariance matrix based on priors, views and beliefs
    # This could be objective function for min variance
    inv_sigma = np.linalg.inv(sigma)
    modified_omega = np.linalg.inv(inv_sigma + np.linalg.inv(omega))
    modified_pi = np.dot(inv_sigma, mu) + np.dot(np.linalg.inv(omega), pi)
    modified_mu = tau * np.dot(sigma, modified_pi)
    modified_sigma = sigma + tau * sigma - tau * np.dot(sigma, np.dot(modified_omega, sigma))

    # Portfolio variance based on modified sigma
    portfolio_variance = np.dot(weights,np.dot(modified_sigma,weights))

    # Objective function 2: Maximize profits
    objective = cp.Maximize(np.matmul(modified_mu, weights) - tau * portfolio_variance)


    # Problem defined and solved
    problem = cp.Problem(objective, constraints)
    problem.solve()
    # weights variable will have optimized weights 
    optimal_weights = weights.value

    # Creating Efficient frontier
    # Generate a range of target returns
    target_returns = np.linspace(min(modified_mu), max(modified_mu), num=100)

    portfolio_weights = []
    portfolio_returns =[]
    portfolio_stds=[]
    for target_return in target_returns:
        # Constraint: Target return should be achieved
        constraints.append(np.sum(modified_mu * weights.value) - tau * portfolio_variance >= target_return)
        np.sum(modified_mu * weights.value)
        # Objective function 1: Minimize the portfolio variance
        objective = cp.Minimize(portfolio_variance)

        # Solve the problem with the additional constraint
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)
        if weights.value is None:
            continue
        else:
            # Store the optimal portfolio weights
            portfolio_weights.append(weights.value)
            portfolio_stds.append(np.sqrt(np.dot(weights.value, np.dot(sigma, weights.value.T))))
            portfolio_returns.append(np.sum(mu * weights.value))
       
    # Convert portfolios to a numpy array
    portfolio_weights = np.array(portfolio_weights)
    portfolio_stds = np.array(portfolio_stds)
    portfolio_returns = np.array(portfolio_returns)

    # Plot the efficient frontier
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_stds, portfolio_returns, marker='o', color='blue', label='Efficient Frontier')
    plt.plot(portfolio_stds, portfolio_returns, color='magenta', linestyle='--', label='Efficient Frontier Line')
    plt.xlabel('Portfolio Standard Deviation')
    plt.ylabel('Portfolio Return')
    plt.title('Black-Litterman Efficient Frontier')
    plt.legend()
    plt.grid(True)
    plt.show()

    return optimal_weights

# Fuction for Critical Line Method for Efficient Frontier
@log_exceptions_warnings
def critcal_line_algo_opt(returns, target=0.15):

    logger.info(f"Running Critical Line Method for Efficient Frontier")


    # expected returns
    mean_daily_returns = returns.mean()
    mu = (1 + mean_daily_returns) ** 252 - 1
    mu = mu.values
    # covariance
    covar = returns.cov()
    sigma = covar*252
    sigma = sigma.values

    def optimal_weights(target_return):
        
        optimal_weights = None
        # Number of assets
        n_assets = len(mu)

        # Variable for the portfolio weights
        weights = cp.Variable(n_assets)

        # Variable representing assets present in portfolio
        z = cp.Variable(n_assets)

        # Variable representing budget constraint in portfolio
        gamma = cp.Parameter(nonneg=True, shape=n_assets)

        # Constraint: All weights are positive
        constraints = [weights >= 0]

        # Constraint: Weights should sum to one
        constraints.append(cp.sum(weights) == 1)

        # Constraint : All asset represents are positive
        constraints.append(z>=0)

        # Constraint : All asset represents sum should be less than one
        constraints.append(cp.sum(z) <= 1)

        # Constraint : Budget constraint
        np.dot(weights,np.dot(sigma, weights))
        constraints.append(np.dot(weights,np.dot(sigma, weights)) <= gamma)
        
        # Constraint : Budget constraint
        constraints.append(weights.T @ mu - target_return == z.T @ mu)

        # Objective function
        objective = cp.Maximize(z.T @ mu)

        while True:
            # Assign a values to the parameter gamma
            # gamma_value = np.ones(n_assets)
            gamma_value = np.random.rand(n_assets)
            gamma.value = gamma_value

            # Problem definition
            problem = cp.Problem(objective, constraints)

            # Solve the problem
            problem.solve(solver=cp.ECOS)
            if weights.value is not None:
                break
        
        # Get the optimal weights
        
        optimal_weights = weights.value

        return optimal_weights

    target_optimal_weights = optimal_weights(target_return=target)

    # Creating Efficient Frontier
    target_returns = np.linspace(mu.min(), mu.max(), num=100)
    portfolio_stds=[]
    portfolio_returns =[]
    portfolio_weights = []

    for target_return in target_returns:
        
        weights = optimal_weights(target_return)

        if weights is None:
            continue
        else:
            # Store the optimal portfolio weights
            portfolio_weights.append(weights)
            portfolio_stds.append(np.sqrt(np.dot(weights, np.dot(sigma, weights.T))))
            portfolio_returns.append(np.sum(mu * weights))
        
     # Convert portfolios to a numpy array
    portfolio_weights = np.array(portfolio_weights)
    portfolio_stds = np.array(portfolio_stds)
    portfolio_returns = np.array(portfolio_returns)
    portfolio_return = target_optimal_weights.dot(mu)
    portfolio_volatility = np.sqrt(target_optimal_weights.dot(sigma).dot(target_optimal_weights))

    # Plot the efficient frontier
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_stds, portfolio_returns, marker='o', color='blue', label='Efficient Frontier')
    #plt.plot(portfolio_stds, portfolio_returns, color='magenta', linestyle='--', label='Efficient Frontier Line')
    #plt.scatter(portfolio_volatility, portfolio_return, marker='o', color='red', label='Target Portfolio')
    plt.xlabel('Portfolio Standard Deviation')
    plt.ylabel('Portfolio Return')
    plt.title('CLT Efficient Frontier')
    plt.legend()
    plt.grid(True)
    plt.show()

    return optimal_weights


# Fuction for Hierarchical Parity Method for Efficient Frontier
@log_exceptions_warnings
def hierarchical_risk_parity_opt(returns):
    logger.info(f"Running Hierarchical Parity Method for Efficient Frontier")

    # expected returns
    mean_daily_returns = returns.mean()
    mu = (1 + mean_daily_returns) ** 252 - 1
    mu = mu.values
    # covariance
    covar = returns.cov()
    sigma = covar*252
    sigma = sigma.values

    # correlation matrix
    correlation_matrix = returns.corr()

    # inverse variance matrix
    inv_var_matrix = np.diag(1 / np.diag(correlation_matrix))

    # distance matrix
    distance_matrix = np.sqrt((1 - correlation_matrix) / 2)

    #linkage matrix using any of linkage types(ex:single, ward)
    linkage_matrix = hierarchy.linkage(distance_matrix, method='ward')

    # Reorder the correlation matrix
    order = hierarchy.leaves_list(linkage_matrix)
    ordered_corr_matrix = correlation_matrix.iloc[order, order]

    # the inverse variance weights
    inv_var_weights = np.sqrt(np.diag(inv_var_matrix))

    # hierarchical risk parity weights
    hrp_weights = inv_var_weights / np.sum(inv_var_weights)

    # modified covariance matrix
    modif_covariance = pd.DataFrame(np.outer(hrp_weights, hrp_weights) * ordered_corr_matrix.values,
                                  index=ordered_corr_matrix.index, columns=ordered_corr_matrix.columns)
    
    # Obtaining weights using this modified covariance
    inv_covariance = pd.DataFrame(np.linalg.pinv(modif_covariance.values), modif_covariance.columns, modif_covariance.index)
    optimal_weights = inv_covariance.sum(axis=1) / inv_covariance.sum().sum()

    # Number of assets
    n_assets = len(mu)

    # Variable for the portfolio weights
    weights = cp.Variable(n_assets)

    # Constraint: All weights are positive
    constraints = [weights >= 0]

    # Constraint: Weights should sum to one
    constraints.append(cp.sum(weights) == 1)
   
    # objective function
    objective = cp.Minimize(np.dot(weights,np.dot(modif_covariance, weights)))
    # Efficient Frontier
    target_returns = np.linspace(mu.min(), mu.max(), num=100)
    portfolio_stds=[]
    portfolio_returns =[]
    portfolio_weights = []

    for target_return in target_returns:
        
        # Constraint: Target return should be achieved
        constraints.append(mu @ weights >= target_return)

        # problem definition
        problem = cp.Problem(objective, constraints)

        # Solve the problem
        problem.solve(solver=cp.ECOS)

        if weights is None:
            continue
        else:
            # Store the optimal portfolio weights
            portfolio_weights.append(weights.value)
            portfolio_stds.append(np.sqrt(np.dot(weights.value, np.dot(sigma, weights.value.T))))
            portfolio_returns.append(np.sum(mu * weights.value))
        
         
    # Convert portfolios to a numpy array
    portfolio_weights = np.array(portfolio_weights)
    portfolio_stds = np.array(portfolio_stds)
    portfolio_returns = np.array(portfolio_returns)
    portfolio_return = optimal_weights.dot(mu)
    portfolio_volatility = np.sqrt(optimal_weights.dot(sigma).dot(optimal_weights))

    # Plot the efficient frontier
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_stds, portfolio_returns, marker='o', color='blue', label='Efficient Frontier')
    plt.plot(portfolio_stds, target_returns, color='magenta', linestyle='--', label='Efficient Frontier Line')
    plt.scatter(portfolio_volatility, portfolio_return, marker='o', color='red', label='Optimal HRP Portfolio')
    plt.xlabel('Portfolio Standard Deviation')
    plt.ylabel('Portfolio Return')
    plt.title('HRP Efficient Frontier')
    plt.legend()
    plt.grid(True)
    plt.show()

    return optimal_weights

        
# Main function
def main():
    try:
        # Load data
        # df = load_data(file_path)
        symbols = get_top_100_stocks()  # Stock symbols
        symbols = ["MSFT","TSLA","AMD","NVDA","META"]
        start_date = '2020-01-01'  # Start date of historical data
        end_date = '2021-01-01'  # End date of historical data
        df = fetch_stock_data(symbols, start_date, end_date)
        # Handle missing data
        df = handle_missing_data(df)
        df = calculate_daily_returns(df)
        returns = df.copy()

       
        try:
            critcal_line_algo_opt(returns, target=0.15)
        except Exception as e:
            logger.exception(f"Exception occurred during  critcal_line script execution: {e}")

        try:
            black_litterman_opt(returns,tau=0.5)
        except Exception as e:
            logger.exception(f"Exception occurred during  black_litterman script execution: {e}")

        try:
            montecarlo_port_opt(returns)
        except Exception as e:
            logger.exception(f"Exception occurred during  montecarlo script execution: {e}")
        
        try:
            hierarchical_risk_parity_opt(returns)
        except Exception as e:
            logger.exception(f"Exception occurred during  hierarchical_risk_parity script execution: {e}")


    except Exception as e:
        logger.exception(f"Exception occurred during script execution: {e}")


# Execute the main function
if __name__ == "__main__":
    main()



