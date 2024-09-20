import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import time
from sklearn.preprocessing import StandardScaler
import sqlite3
import pandas as pd
import yfinance as yf
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.data.historical import StockHistoricalDataClient
from datetime import datetime
import config  
import matplotlib.pyplot as plt

class SimpleStockTrainingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data, initial_balance=1000000):
        super(SimpleStockTrainingEnv, self).__init__()
        
        # Data
        print(data.shape)
        self.data = data
        if self.data.empty:
            raise ValueError("DataFrame is empty. Cannot reset environment.")
        self.initial_balance = initial_balance
        self.n_step = self.data.shape[0]
        self.stock_buy_ratio  = 0.05
        self.stock_sell_ratio = 0.05

        # Action space: Buy(5%, 10%, 20%,30%,40%, 50%, 60%, 70%, 80%, 90%, 100% of available cash), Sell, Hold
        self.action_space = spaces.Discrete(3)

        # State size contains all indicators and balance and shares held
        self.state_size = self.data.shape[1] + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_size,))
        

        self.action_history = []
        self.price_history = []
        self.total_asset = []
        # Start the first episode
        self.reset()

    def step(self, action):
        # Execute one time step within the environment
        self.current_step += 1
        self.prev_total = self.balance + self.current_price * self.shares_held
        # Assume self._take_action implements the action and updates the balance
        self._take_action(action)
    
        
        done = self.current_step >= self.n_step - 1
        reward = self._calculate_reward()
        
        self.action_history.append(action)
        self.price_history.append(self.current_price)
        self.total_asset.append(self.balance + self.current_price * self.shares_held)
        # Set placeholder for info
        info = {}
        # if done:
        #     self.reset()
        # Return step information
        return self.state, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        if self.data.empty:
            raise ValueError("DataFrame is empty. Cannot reset environment.")
        self.current_price = self.data.iloc[0]['current_price']
        self.balance = self.initial_balance
        self.prev_total = self.initial_balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.action_history = []
        self.price_history = []
        self.total_asset = []
        
        # Set the current state
        self.state = self._get_state()
        
        return self.state

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Total sales value: {self.total_sales_value}')
        # ... additional prints can show more info about the last step
        
    def _get_state(self):
        # Combine data points from the dataframe with balance and shares held
        frame = np.append(self.data.iloc[self.current_step ].values, [self.balance, self.shares_held])
        return frame
    
    def _take_action(self, action):
        # Determine action and update balance
        self.current_price = self.data.iloc[self.current_step]['current_price']

        if action == 1: 
            self.stock_buy_ratio = 0.1 
        elif action == 2: 
            self.stock_sell_ratio = 0.1

        if action == 1 :
            self._buy_stock(self.current_price)
        if action == 2: 
            self._sell_stock(self.current_price)
            
        
    def _calculate_reward(self):
        # Calculate reward, typically change in portfolio value
        return self.balance + self.shares_held * self.current_price - self.prev_total

    def _buy_stock(self, current_price):

        total_possible = int(self.balance / current_price)
        shares_to_buy = int(total_possible * self.stock_buy_ratio)
        cost = shares_to_buy * current_price

        if cost <= self.balance:
            self.balance -= cost
            self.shares_held += shares_to_buy
        else:
            shares_to_buy = total_possible
            cost = shares_to_buy * current_price
            if cost <= self.balance:
                self.balance -= cost
                self.shares_held += shares_to_buy
        
    
    def _sell_stock(self, current_price):
        if self.shares_held > 0:
            sell_amount =  int(self.shares_held * self.stock_sell_ratio)
            # Sell all the shares
            self.balance += sell_amount * current_price
            self.total_shares_sold += sell_amount
            self.total_sales_value += sell_amount * current_price
            self.shares_held = self.shares_held - sell_amount  # Reset shares held to 0 after selling
        else:
            # If you don't hold any shares, you can't sell
            pass  # Do nothing, or
            # print("No shares to sell.")
        # Update balance and shares held for selling

    @property
    def total_episodes(self): 
        return len(self.data)

class DiverseStockTrainingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data, initial_balance=1000000, shares = 0):
        super(DiverseStockTrainingEnv, self).__init__()
        
        # Data

        self.data = data
        self.initial_balance = initial_balance
        self.shares = shares
      # Validate if data is provided
        if self.data is not None and self.data.empty:
            raise ValueError("DataFrame is empty. Cannot reset environment.")
        

        self.n_step = self.data.shape[0] if self.data is not None else 0
        self.stock_buy_ratio  = 0.05
        self.stock_sell_ratio = 0.05

        # Action space: Buy(5%, 10%, 20%,30%,40%, 50%, 60%, 70%, 80%, 90%, 100% of available cash), Sell, Hold
        self.action_space = spaces.Discrete(23)

        if self.data is not None:
            self.state_size = self.data.shape[1] + 2
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_size,))
        else:
            self.state_size = None
            self.observation_space = None

        self.action_history = []
        self.price_history = []
        self.total_asset = []

        if self.data is not None:
            self.reset()  # Only reset if data is provided

    def step(self, action):
        if self.data is None or self.n_step == 0:
            raise ValueError("No data provided. Cannot execute step.")
        # Execute one time step within the environment
        self.current_step += 1
        self.prev_total = self.balance + self.current_price * self.shares_held
        # Assume self._take_action implements the action and updates the balance
        self._take_action(action)
    
        
        done = self.current_step >= self.n_step - 1
        reward = self._calculate_reward()
        
        self.action_history.append(action)
        self.price_history.append(self.current_price)
        self.total_asset.append(self.balance + self.current_price * self.shares_held)
        # Set placeholder for info
        info = {}
        # if done:
        #     self.reset()
        # Return step information
        return self.state, reward, done, info

    def reset(self):

        if self.data is None or self.data.empty:
            raise ValueError("No data provided. Cannot reset environment.")
        # Reset the state of the environment to an initial state
        self.current_step = 0
        if self.data.empty:
            raise ValueError("DataFrame is empty. Cannot reset environment.")
        self.current_price = self.data.iloc[0]['current_price']
        self.balance = self.initial_balance
        self.prev_total = self.initial_balance
        self.shares_held = self.shares
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.action_history = []
        self.price_history = []
        self.total_asset = []
        
        # Set the current state
        self.state = self._get_state()
        
        return self.state

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Total sales value: {self.total_sales_value}')
        # ... additional prints can show more info about the last step
        
    def _get_state(self):
        if self.data is None or self.data.empty:
            raise ValueError("No data provided. Cannot retrieve state.")
        # Combine data points from the dataframe with balance and shares held
        frame = np.append(self.data.iloc[self.current_step ].values, [self.balance, self.shares_held])
        print('frame shape : ' , frame.shape)
        print('frame' , frame)
        return frame
    
    def _take_action(self, action):
        if self.data is None or self.data.empty:
            raise ValueError("No data provided. Cannot take action.")
        
        # Determine action and update balance
        self.current_price = self.data.iloc[self.current_step]['current_price']

        # Determine the buy/sell ratio based on the action
        if 1 <= action <= 11:  # Buy actions
            self.stock_buy_ratio = (action) / 20  # Ranges from 0.05 to 0.55
            self._buy_stock(self.current_price)
        elif 12 <= action <= 22:  # Sell actions
            self.stock_sell_ratio = (action - 11) / 20  # Ranges from 0.05 to 0.55
            self._sell_stock(self.current_price)
            
        
    def _calculate_reward(self):
        # Calculate reward, typically change in portfolio value
        return self.balance + self.shares_held * self.current_price - self.prev_total

    def _buy_stock(self, current_price):

        total_possible = int(self.balance / current_price)
        shares_to_buy = int(total_possible * self.stock_buy_ratio)
        cost = shares_to_buy * current_price

        if cost <= self.balance:
            self.balance -= cost
            self.shares_held += shares_to_buy
    
    def _sell_stock(self, current_price):
        if self.shares_held > 0:
            sell_amount =  int(self.shares_held * self.stock_sell_ratio)
            # Sell all the shares
            self.balance += sell_amount * current_price
            self.total_shares_sold += sell_amount
            self.total_sales_value += sell_amount * current_price
            self.shares_held = self.shares_held - sell_amount  # Reset shares held to 0 after selling
        else:
            # If you don't hold any shares, you can't sell
            pass  # Do nothing, or
            # print("No shares to sell.")
        # Update balance and shares held for selling

    @property
    def total_episodes(self):         
        if self.data is None:
            raise ValueError("No data provided. Cannot determine total episodes.")
        return len(self.data)
    
class TimeCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TimeCallback, self).__init__(verbose)
        self.start_time = time.time()

    def _on_step(self):
        # Calculate elapsed time and remaining time
        elapsed_time = time.time() - self.start_time
        total_timesteps = self.locals['total_timesteps']
        current_timesteps = self.num_timesteps
        if current_timesteps == 0:
            return True  # Avoid division by zero

        estimated_total_time = elapsed_time * (total_timesteps / current_timesteps)
        estimated_time_left = estimated_total_time - elapsed_time

        # Convert seconds to a more readable format
        remaining_time = time.strftime("%H:%M:%S", time.gmtime(estimated_time_left))
        
        # Print progress and time left every 1000 steps
        if current_timesteps % 1000 == 0:
            print(f"Step: {current_timesteps}/{total_timesteps} - Remaining Time: {remaining_time}")

        return True
    
# Fetch latest quote (bid/ask prices) from Alpaca
def get_latest_quote(symbol):
    data_client = StockHistoricalDataClient(config.API_KEY, config.SECRETE_KEY)
    request_params = StockLatestQuoteRequest(
        symbol_or_symbols=symbol
    )
    quote = data_client.get_stock_latest_quote(request_params)
    return quote

# Fetch current price and market cap from Yahoo Finance
def get_yahoo_data(symbol):
    stock = yf.Ticker(symbol)
    info = stock.info
    return (
        info.get('currentPrice'),
        info.get('marketCap'),
        info.get('enterpriseValue'),
        info.get('trailingPE'),
        info.get('forwardPE'),
        info.get('pegRatio'),
        info.get('priceToSalesTrailing12Months'),
        info.get('priceToBook'),
        info.get('enterpriseToRevenue'),
        info.get('enterpriseToEbitda'),
        info.get('totalRevenue'),
        info.get('grossProfits'),
        info.get('ebitda'),
        info.get('netIncomeToCommon'),
        info.get('beta'),
        info.get('fiftyTwoWeekHigh'),
        info.get('fiftyTwoWeekLow'),
        info.get('fiftyDayAverage'),
        info.get('twoHundredDayAverage'),
        info.get('dividendYield'),
        info.get('payoutRatio'),
        info.get('sector')
    )

# Function to store the latest stock data in the database
def store_latest_stock_data(symbol, conn, cursor):
    try:
        # Get data from Alpaca
        quote = get_latest_quote(symbol)
        bid_price = quote[symbol].bid_price
        ask_price = quote[symbol].ask_price
        bid_size = quote[symbol].bid_size
        ask_size = quote[symbol].ask_size
        timestamp = quote[symbol].timestamp

        # Get data from Yahoo Finance
        yahoo_data = get_yahoo_data(symbol)

        # Get the current retrieval time
        retrieval_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Insert data into the database
        insert_query = '''
        INSERT INTO latest_stock_data (
            symbol, bid_price, ask_price, bid_size, ask_size, current_price, market_cap, 
            enterprise_value, trailing_pe, forward_pe, peg_ratio, price_to_sales, price_to_book, 
            enterprise_to_revenue, enterprise_to_ebitda, total_revenue, gross_profits, ebitda, 
            net_income, beta, fifty_two_week_high, fifty_two_week_low, fifty_day_average, 
            two_hundred_day_average, dividend_yield, payout_ratio, sector, timestamp, retrieval_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        '''
        cursor.execute(insert_query, (
            symbol, bid_price, ask_price, bid_size, ask_size, *yahoo_data, timestamp, retrieval_time
        ))
        conn.commit()
    except Exception as e:
        print(f"Error storing data for {symbol}: {e}")


def preprocess_latest_stock_data(df):
    reference_columns = ['current_price', 'market_cap', 'enterprise_value', 'trailing_pe',
       'forward_pe', 'peg_ratio', 'price_to_sales', 'price_to_book',
       'enterprise_to_revenue', 'enterprise_to_ebitda', 'total_revenue',
       'ebitda', 'net_income', 'beta', 'fifty_two_week_high',
       'fifty_two_week_low', 'fifty_day_average', 'two_hundred_day_average',
       'dividend_yield', 'CPI', 'UNEMPLOYMENT', 'FEDERAL_FUNDS_RATE',
       'RETAIL_SALES', 'DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3',
       'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30', 'DXY', 'market_cap_missing',
       'enterprise_value_missing', 'trailing_pe_missing', 'forward_pe_missing',
       'peg_ratio_missing', 'price_to_sales_missing', 'price_to_book_missing',
       'enterprise_to_revenue_missing', 'enterprise_to_ebitda_missing',
       'total_revenue_missing', 'ebitda_missing', 'net_income_missing',
       'beta_missing', 'fifty_two_week_high_missing',
       'fifty_two_week_low_missing', 'fifty_day_average_missing',
       'two_hundred_day_average_missing', 'dividend_yield_missing',
       'sector_missing', 'CPI_missing', 'UNEMPLOYMENT_missing',
       'FEDERAL_FUNDS_RATE_missing', 'RETAIL_SALES_missing', 'DGS1MO_missing',
       'DGS3MO_missing', 'DGS6MO_missing', 'DGS1_missing', 'DGS2_missing',
       'DGS3_missing', 'DGS5_missing', 'DGS7_missing', 'DGS10_missing',
       'DGS20_missing', 'DGS30_missing', 'DXY_missing', 'sector_0',
       'sector_Basic Materials', 'sector_Communication Services',
       'sector_Consumer Cyclical', 'sector_Consumer Defensive',
       'sector_Energy', 'sector_Financial Services', 'sector_Healthcare',
       'sector_Industrials', 'sector_Real Estate', 'sector_Technology',
       'sector_Utilities']
    
    # Extract relevant features
    features = df[[ 'current_price', 'market_cap', 
                   'enterprise_value', 'trailing_pe', 'forward_pe', 'peg_ratio', 
                   'price_to_sales', 'price_to_book', 'enterprise_to_revenue', 
                   'enterprise_to_ebitda', 'total_revenue', 
                   'ebitda', 'net_income', 'beta', 'fifty_two_week_high', 
                   'fifty_two_week_low', 'fifty_day_average', 
                   'two_hundred_day_average', 'dividend_yield', 'sector', 'CPI', 
                   'UNEMPLOYMENT', 'FEDERAL_FUNDS_RATE', 'RETAIL_SALES', 'DGS1MO', 
                   'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30', 'DXY']]
    print(features)
    # Create missing indicators for all columns
    for col in features.columns:
        if col != 'current_price':
        # Check if there are any missing values
            features[f'{col}_missing'] = features[col].isnull().astype(int)

    # Fill missing values with a placeholder (e.g., 0)
    features.fillna(0, inplace=True)

    # One-Hot Encode the 'sector' column
    features = pd.get_dummies(features, columns=['sector'])

        # Ensure all possible sector columns are present
    all_possible_sectors = ['sector_Basic Materials', 'sector_Communication Services', 'sector_Consumer Cyclical',
                            'sector_Consumer Defensive', 'sector_Energy', 'sector_Financial Services', 
                            'sector_Healthcare', 'sector_Industrials', 'sector_Real Estate', 
                            'sector_Technology', 'sector_Utilities']
    
    for sector in all_possible_sectors:
        if sector not in features.columns:
            features[sector] = 0  # Add the missing sector with zeros
    
    # Ensure that all expected columns are present and in the same order as the reference DataFrame
    for col in reference_columns:
        if col not in features.columns:
            features[col] = 0 

    # Normalize the features
    # scaler = StandardScaler()
    # features = scaler.fit_transform(features)
    
    features = features[reference_columns]
    print(features.columns)
    return features

def fetch_data_from_db(db_path, query):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    
    # Execute the query and fetch the data
    df = pd.read_sql_query(query, conn)
    
    # Close the connection
    conn.close()
    
    return df

def combine_latest_to_econ():
    # Connect to your database
    conn = sqlite3.connect('stock.db')

    # Combine the data using the SQL query
    query = '''
    SELECT 
        a.symbol,
        a.bid_price,
        a.ask_price,
        a.current_price,
        a.market_cap,
        a.enterprise_value,
        a.trailing_pe,
        a.forward_pe,
        a.peg_ratio,
        a.price_to_sales,
        a.price_to_book,
        a.enterprise_to_revenue,
        a.enterprise_to_ebitda,
        a.total_revenue,
        a.gross_profits,
        a.ebitda,
        a.net_income,
        a.beta,
        a.fifty_two_week_high,
        a.fifty_two_week_low,
        a.fifty_day_average,
        a.two_hundred_day_average,
        a.dividend_yield,
        a.sector,
        a.timestamp,
        c.CPI,
        c.UNEMPLOYMENT,
        c.FEDERAL_FUNDS_RATE,
        c.RETAIL_SALES,
        c.DGS1MO,
        c.DGS3MO,
        c.DGS6MO,
        c.DGS1,
        c.DGS2,
        c.DGS3,
        c.DGS5,
        c.DGS7,
        c.DGS10,
        c.DGS20,
        c.DGS30,
        c.DXY
    FROM 
        latest_stock_data a
    LEFT JOIN 
        combined_economic_treasury_data c
    ON 
        c.date = (SELECT max(c2.date)
                FROM combined_economic_treasury_data c2
                WHERE c2.date < date(a.timestamp))
    '''

    # Execute the query and fetch the result into a DataFrame
    combined_data = pd.read_sql_query(query, conn)

    # If you want to save this combined data into a new table
    combined_data.to_sql('final_combined_latest_data', conn, if_exists='replace', index=False)

    # Close the connection
    conn.close()

    # Display the first few rows of the combined data
    print(combined_data.head())

def show_graph_diverse(env, save_path = None):
    # Assuming 'env' is your environment instance after a simulation run
    prices = env.price_history
    actions = env.action_history
    portfolio_values = env.total_asset
    n = 50 # Adjust n based on the total number of points and desired detail
    prices_downsampled = prices[::n]
    portfolio_values_downsampled = portfolio_values[::n]
    actions_downsampled = actions[::n]

    fig, ax1 = plt.subplots()

    # Plotting the stock prices
    color = 'tab:red'
    ax1.set_xlabel('Time (steps)')
    ax1.set_ylabel('Price', color=color)
    ax1.plot(prices_downsampled, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a twin axis to plot portfolio value
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Portfolio Value', color=color)
    ax2.plot(portfolio_values_downsampled, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Mark buy and sell actions on the graph
    for i, action in enumerate(actions_downsampled):
        if 1 <= action and action <= 11:  
            ax1.scatter(i, prices[i], color='green', label='buy', marker='^', alpha=0.7 )
        elif 12 <= action and  action <= 22: 
            ax1.scatter(i, prices[i], color='red', label='sell', marker='v', alpha=0.7)

    # Adding legend with unique entries
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # removing duplicates in legend
    ax1.legend(by_label.values(), by_label.keys())

    plt.title('Stock Prices, Buy/Sell Actions and Portfolio Value')
    if save_path:
        plt.savefig(save_path)
        print(f"Graph saved to {save_path}")


def simple_show_graph(env, save_path = None):
    # Assuming 'env' is your environment instance after a simulation run
    prices = env.price_history
    actions = env.action_history
    portfolio_values = env.total_asset
    n = 50 # Adjust n based on the total number of points and desired detail
    prices_downsampled = prices[::n]
    portfolio_values_downsampled = portfolio_values[::n]
    actions_downsampled = actions[::n]

    fig, ax1 = plt.subplots()

    # Plotting the stock prices
    color = 'tab:red'
    ax1.set_xlabel('Time (steps)')
    ax1.set_ylabel('Price', color=color)
    ax1.plot(prices_downsampled, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a twin axis to plot portfolio value
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Portfolio Value', color=color)
    ax2.plot(portfolio_values_downsampled, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Mark buy and sell actions on the graph
    for i, action in enumerate(actions_downsampled):
        if action == 1:  
            ax1.scatter(i, prices[i], color='green', label='buy', marker='^', alpha=0.7 )
        elif action == 2: 
            ax1.scatter(i, prices[i], color='red', label='sell', marker='v', alpha=0.7)

    # Adding legend with unique entries
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # removing duplicates in legend
    ax1.legend(by_label.values(), by_label.keys())

    plt.title('Stock Prices, Buy/Sell Actions and Portfolio Value')
    if save_path:
        plt.savefig(save_path)
        print(f"Graph saved to {save_path}")


