import schedule
import time
from datetime import datetime
import store_econ_data
import config
import fetch_dqn
from stable_baselines3 import DQN
import store_companies
import mlp
import torch 
import store_data
import config
from alpaca_trade_api.rest import REST, TimeFrame
import numpy as np

from alpaca_trade_api.rest import APIError

class LiveTrain:

    def __init__(self, symbol = 'AAPL'):
        
        store_companies.store_latest_companies_data()
        self.fetch_treasury_and_econ_data()
        self.stock_data = mlp.retrieve_final_combined_data()
        self.input_size, self.hidden_size, self.output_size, self.mlp_pth = mlp.train_and_save_mlp(stock_data= self.stock_data)
        self.symbol = symbol
        self.alpaca_api = REST(config.API_KEY, config.SECRETE_KEY, base_url= 'https://paper-api.alpaca.markets')
        self.dqn_name = 'stock_dqn_model'
        self.dqn_pth = self.dqn_name + '.zip'
                # Define market hours in Mountain Time
        self.market_open = datetime.now().replace(hour=7, minute=30, second=0, microsecond=0)
        self.market_close = datetime.now().replace(hour=14, minute=0, second=0, microsecond=0)

    def combine_mlp_prediction(self, predictions, stock_data):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().numpy()
        
        print(predictions.shape)
        # Ensure predictions are 1D before adding them as a column
        if predictions.ndim > 1:
            # Select the specific column corresponding to `current_price`
            predictions = predictions[:, 0]  # Assuming the `current_price` is the first column in predictions
        
        print(predictions)
        # Add predictions to the DataFrame
        stock_data['mlp_predictions'] = predictions

        return stock_data
    
    def is_weekday(self):
        current_day = datetime.now().weekday()
        return current_day < 5  # 0 = Monday, 4 = Friday, 5 = Saturday, 6 = Sunday

    def fetch_treasury_and_econ_data(self):
        if self.is_weekday():
            print("Fetching Treasury Yield and Economic data...")

            # Econ data
            economic_data = store_econ_data.fetch_all_economic_data()
            store_econ_data.drop_table(db_name='stock.db', table_name='economic_history_data')
            # Save the economic data to the SQLite database
            store_econ_data.save_to_sqlite_econ(economic_data)

            # Example usage: Get multiple treasury yields and store them in the database
            api_key = config.FRED_API_KEY  # Replace with your FRED API key
            start_date = '1776-07-04'
            end_date = '9999-12-31'
            # Fetch the data
            treasury_yields = store_econ_data.fetch_treasury_yields(api_key, start_date, end_date)
            store_econ_data.drop_table('stock.db','treasury_yields')

            # Save the data to the SQLite database
            store_econ_data.save_to_sqlite_treasury(treasury_yields)

            store_econ_data.add_dxy_to_treasury()
            store_econ_data.combine_economic_treasury_data()

            # create latest_stock_data table if not exists
            store_econ_data.create_latest_stock_data()
            # combine latest_stock_data with econ data -> results final_combined_latest_data
            store_econ_data.combine_latest_to_econ()

            # Your code here
        else:
            print("It's the weekend. Skipping Treasury Yield and Economic data fetch.")

    def fetch_and_train_dqn(self):
        if self.is_weekday():
            print("Fetching Apple data and training DQN model...")
            current_time = datetime.now()
            model = DQN.load(self.dqn_pth)

            current_time = datetime.now()

            while self.market_open <= current_time and current_time <= self.market_close and current_time.weekday() < 5:

                live_state = self.get_live_state()
                action, _states = model.predict(live_state)

                print('live_State: ' ,live_state)
                print('action: ', action)
                print('after state: ', _states)

                    # Execute the action using the Alpaca API
                if action >= 1 and action <= 11:  # Buy actions
                    # Calculate buy amount based on the action
                    amount_to_buy = self.calculate_buy_amount(action, live_state[0][0])
                    self.alpaca_api.submit_order(symbol=self.symbol, qty=amount_to_buy, side='buy', type='market', time_in_force='gtc')
                    print(f"Bought {amount_to_buy} of {self.symbol} at {live_state[0][0]}")

                elif action >= 12 and action <= 22:  # Sell actions
                    # Calculate sell amount based on the action
                    amount_to_sell = self.calculate_sell_amount(action)
                    self.alpaca_api.submit_order(symbol=self.symbol, qty=amount_to_sell, side='sell', type='market', time_in_force='gtc')
                    print(f"Sold {amount_to_sell} of {self.symbol} at {live_state[0][0]}")

                    # Optional: Sleep for a certain amount of time before making the next decision

                time.sleep(10)  # For example, wait 1 minute between trades
                current_time = datetime.now()
        else:
            print("It's the weekend. Skipping DQN training.")

    def fetch_and_train_mlp(self):
        if self.is_weekday():
            print("Fetching all companies' data and training MLP model...")
            
            # Your code here
            store_companies.store_latest_companies_data()
            stock_data = mlp.retrieve_final_combined_data()
            self.input_size, self.hidden_size, self.output_size, self.mlp_pth = mlp.train_and_save_mlp(stock_data= stock_data)

        else:
            print("It's the weekend. Skipping MLP training.")

    def keep_program_running(self):
        print("Program is running and waiting for scheduled tasks...")

    def train_dqn_existing(self): 
        db_path = 'stock.db'
        query = f'SELECT * FROM final_combined_latest_data WHERE symbol = "{self.symbol}" ORDER BY timestamp ASC' 
        df = fetch_dqn.fetch_data_from_db(db_path, query)

        stock_data = fetch_dqn.preprocess_latest_stock_data(df)
        combined_data = self.add_predictions_to_data(stock_data= stock_data)

        print('combined data (added predictions) shape: ', combined_data.shape)
        print('combined data (added predictions): ', combined_data.head())

        # Instantiate and use the callback
        time_callback = fetch_dqn.TimeCallback()
        # Initialize your custom environment
        env = fetch_dqn.DiverseStockTrainingEnv(data= combined_data)  

        # Configure and instantiate DQN
        model = DQN("MlpPolicy", env, verbose=1, buffer_size=10000, learning_rate=0.01,
                    batch_size=32, gamma=0.99, exploration_fraction=0.1,
                    exploration_final_eps=0.02, target_update_interval=250)

        # Train the model
        model.learn(total_timesteps=len(stock_data)*5-200, callback=time_callback)
        fetch_dqn.show_graph_diverse(env, save_path='my_dqn_diverse_graph.png')
        model.save(self.dqn_name)

    def get_available_cash(self):
        account = self.alpaca_api.get_account()
        return float(account.cash)


    # Define a function to get held shares from Alpaca
    def get_held_shares(self):
        try:
            position = self.alpaca_api.get_position(self.symbol)
            return int(position.qty)
        except APIError as e:
            if 'position does not exist' in str(e):
                print(f"No position found for {self.symbol}. Returning 0 shares.")
                return 0
            else:
                raise  # re-raise the exception if it's something else
    
    def add_predictions_to_data(self, stock_data):
        X = mlp.convert_tensor_compatibility(stock_data)
        predictions = mlp.get_predictions(X, self.input_size, self.hidden_size, self.output_size, self.mlp_pth)
        combined_data = self.combine_mlp_prediction(predictions, stock_data)

        return combined_data 

    def get_live_state(self): 
        new_data_df = store_data.store_and_retrieve_latest_company_data(self.symbol)
        stock_data = fetch_dqn.preprocess_latest_stock_data(new_data_df)
        combined_data = self.add_predictions_to_data(stock_data= stock_data)
        
        balance = self.get_available_cash()
        shares_held = self.get_held_shares()

        live_state = np.append(combined_data, [balance, shares_held])
        live_state = np.array(live_state, dtype=np.float64)
        
        # Ensure it's 2D if needed
        if live_state.ndim == 1:
            live_state = live_state.reshape(1, -1)

        print('live state shape: ', live_state.shape) 
        print('live state: ', live_state)

        return live_state 
    # Define a function to calculate the buy amount
    def calculate_buy_amount(self, action, current_price):
        # Calculate the buy ratio (0.05 to 0.55) based on the action
        buy_ratio = (action[0]) / 20  # Scale the action to get the buy ratio
        available_cash = self.get_available_cash()
        buy_amount = available_cash * buy_ratio / current_price  # Calculate how many shares to buy
        return int(buy_amount)

    def calculate_sell_amount(self, action):
        # Calculate the sell ratio (0.05 to 0.55) based on the action
        sell_ratio = (action[0] - 11) / 20  # Scale the action to get the sell ratio
        held_shares = self.get_held_shares()
        sell_amount = held_shares * sell_ratio  # Calculate how many shares to sell
        return int(sell_amount)

    
    def start(self): 
        # self.train_dqn_existing()
        # self.fetch_and_train_dqn()
        # Schedule tasks
        schedule.every().day.at("06:00").do(self.fetch_treasury_and_econ_data)
        schedule.every().day.at("07:00").do(self.fetch_and_train_dqn)
        schedule.every().day.at("14:00").do(self.fetch_and_train_mlp)

        # Continuous loop
        while True:
            schedule.run_pending()
            self.keep_program_running()
            time.sleep(60)  # Check the schedule every minute


live_train = LiveTrain() 
live_train.start()
