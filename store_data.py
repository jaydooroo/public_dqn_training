from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
import config  # Assuming API_KEY and SECRET_KEY are stored in a config module
import yfinance as yf
import sqlite3
import time
from datetime import datetime, timedelta
import schedule
import pandas as pd

# Fetch latest quote (bid/ask prices) from Alpaca
def get_latest_quote(symbol, data_client):
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
        info.get('payoutRatio')
    )

# Function to store the latest stock data in the database
def store_latest_stock_data(symbol, cursor, conn, data_client):
    try:
        # Get data from Alpaca
        quote = get_latest_quote(symbol, data_client)
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
            two_hundred_day_average, dividend_yield, payout_ratio, timestamp, retrieval_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        '''
        cursor.execute(insert_query, (
            symbol, bid_price, ask_price, bid_size, ask_size, *yahoo_data, timestamp, retrieval_time
        ))
        conn.commit()

        # Retrieve the data you just inserted using the last row ID and pd.read_sql_query
        last_row_id = cursor.lastrowid
        select_query = 'SELECT * FROM latest_stock_data WHERE id = ?;'
        inserted_data_df = pd.read_sql_query(select_query, conn, params=(last_row_id,))

        print("Inserted data:", inserted_data_df)  # Display the inserted data

        return inserted_data_df
    
    except Exception as e:
        print(f"Error storing data for {symbol}: {e}")

def store_all_company_data(symbol = 'AAPL'): 
    # Connect to the SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('stock.db')

    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()

    # Create a table to store the latest stock data if it doesn't exist
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS latest_stock_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        bid_price REAL,
        ask_price REAL,
        bid_size INTEGER,
        ask_size INTEGER,
        current_price REAL,
        market_cap REAL,
        enterprise_value REAL,
        trailing_pe REAL,
        forward_pe REAL,
        peg_ratio REAL,
        price_to_sales REAL,
        price_to_book REAL,
        enterprise_to_revenue REAL,
        enterprise_to_ebitda REAL,
        total_revenue REAL,
        gross_profits REAL,
        ebitda REAL,
        net_income REAL,
        beta REAL,
        fifty_two_week_high REAL,
        fifty_two_week_low REAL,
        fifty_day_average REAL,
        two_hundred_day_average REAL,
        dividend_yield REAL,
        payout_ratio REAL,
        timestamp TEXT,
        retrieval_time TEXT
    );
    '''
    cursor.execute(create_table_query)
    conn.commit()
    # Initialize the historical data client
    data_client = StockHistoricalDataClient(config.API_KEY, config.SECRETE_KEY)

    # Define market hours in Mountain Time
    market_open = datetime.now().replace(hour=7, minute=30, second=0, microsecond=0)
    market_close = datetime.now().replace(hour=14, minute=0, second=0, microsecond=0)
    current_time = datetime.now()

    # Continuous loop to fetch and store data
    while market_open <= current_time and current_time <= market_close and current_time.weekday() < 5:
        store_latest_stock_data(symbol, cursor= cursor, conn= conn, data_client= data_client)
        print('Successfully stored.', ' ', symbol )
        time.sleep(3600 / 1999)  # Approximately 1.8 seconds
        current_time = datetime.now()
    # Close the cursor and connection when done (this won't be reached in an infinite loop)
    cursor.close()
    conn.close()

def store_and_retrieve_latest_company_data(symbol = 'AAPL'): 
    conn = sqlite3.connect('stock.db')
    cursor = conn.cursor()

    # Initialize the historical data client
    data_client = StockHistoricalDataClient(config.API_KEY, config.SECRETE_KEY)


    inserted_data = store_latest_stock_data(symbol, cursor, conn, data_client)
    print('Successfully stored.', symbol)

       # Extract the timestamp from the inserted data
    latest_timestamp = inserted_data['timestamp'].iloc[0]
    latest_date = latest_timestamp.split(' ')[0]  # Extract just the date part

   # Query to get the closest economic data based on the latest_timestamp
    query = f'''
    SELECT * FROM combined_economic_treasury_data
    WHERE date < '{latest_date}'
    ORDER BY date DESC
    LIMIT 1;
    '''
    
    econ_data = pd.read_sql_query(query, conn)
    print('Econ data: ', econ_data)
    print('Econ data shape', econ_data.shape)

    # Combine inserted_data with econ_data based on the closest date
    combined_data = pd.concat([inserted_data, econ_data], axis=1)
    print('Combined Data:', combined_data)
    print('Combined Data Shape:', combined_data.shape)

    cursor.close()
    conn.close()

    return combined_data
