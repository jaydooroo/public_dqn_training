import yfinance as yf
import sqlite3
import time
from datetime import datetime
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
import config  # Assuming API_KEY and SECRET_KEY are stored in a config module

# Fetch latest quote (bid/ask prices) from Alpaca
def get_latest_quote(symbol, data_client):
    try:
        request_params = StockLatestQuoteRequest(
            symbol_or_symbols=symbol
        )
        quote = data_client.get_stock_latest_quote(request_params)
        return quote
    except Exception as e:
        print(f"Error fetching Alpaca data for {symbol}: {e}")
        return None

# Fetch current price, market cap, and sector from Yahoo Finance
def get_yahoo_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        if 'currentPrice' not in info:
            print(f"Ticker {symbol} does not have complete Yahoo data.")
            return (None,) * 20  # Return a tuple of Nones if data is missing

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
            info.get('sector')  # Added sector here
        )
    except Exception as e:
        print(f"Error fetching Yahoo data for {symbol}: {e}")
        return (None,) * 20  # Return a tuple of Nones if there's an error

# Function to store the latest stock data in the database
def store_latest_stock_data(symbol, cursor, conn, data_client):
    try:
        # Get data from Alpaca
        quote = get_latest_quote(symbol, data_client)
        
        if quote is None or symbol not in quote:
            print(f"No Alpaca data for {symbol}. Skipping...")
            return

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
        INSERT INTO all_companies_data (
            symbol, bid_price, ask_price, bid_size, ask_size, current_price, market_cap, 
            enterprise_value, trailing_pe, forward_pe, peg_ratio, price_to_sales, price_to_book, 
            enterprise_to_revenue, enterprise_to_ebitda, total_revenue, gross_profits, ebitda, 
            net_income, beta, fifty_two_week_high, fifty_two_week_low, fifty_day_average, 
            two_hundred_day_average, dividend_yield, sector, timestamp, retrieval_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        '''
        cursor.execute(insert_query, (
            symbol, bid_price, ask_price, bid_size, ask_size, *yahoo_data, timestamp, retrieval_time
        ))
        conn.commit()

        # Retrieve the data you just inserted using the last row ID
        last_row_id = cursor.lastrowid
        select_query = 'SELECT * FROM latest_stock_data WHERE id = ?;'
        cursor.execute(select_query, (last_row_id,))
        inserted_data = cursor.fetchone()

        print("Inserted data:", inserted_data)  # Display the inserted data

        return inserted_data
    
    except Exception as e:
        print(f"Error storing data for {symbol}: {e}")

# Function to reset the table
def reset_table():
    conn = sqlite3.connect('stock.db')
    cursor = conn.cursor()
    
    cursor.execute('DROP TABLE IF EXISTS all_companies_data')

    # Recreate the table
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS all_companies_data (
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
        sector TEXT,
        timestamp TEXT,
        retrieval_time TEXT
    );
    '''
    cursor.execute(create_table_query)
    conn.commit()
    conn.close()

def store_latest_companies_data(): 

    # Reset the table (uncomment this line if you need to reset the table)
    # reset_table()

    # print("The table 'all_companies_data' has been reset.")

    # Connect to the database
    conn = sqlite3.connect('stock.db')

    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()

    # Initialize the historical data client
    data_client = StockHistoricalDataClient(config.API_KEY, config.SECRETE_KEY)

    conn_ticket = sqlite3.connect('tickers.db')
    cursor_ticket = conn_ticket.cursor()
    cursor_ticket.execute('SELECT ticker FROM symbols')
    # Fetchall returns a list of tuples, each containing a single element
    rows = cursor_ticket.fetchall()

    # Optional: Filter out unsupported tickers
    # unsupported_suffixes = ('.U', '.WS', '.W', '.P', '.F', '.Q', '.R')
    # symbols = [symbol for symbol in symbols if not any(symbol.endswith(suffix) for suffix in unsupported_suffixes)]
    for row in rows:
        # Get the current time
        # Fetch and store data if market is open
        store_latest_stock_data(row[0], cursor, conn, data_client)
        print('Successfully stored.', row[0])

        # Sleep to avoid hitting API limits
        time.sleep(3600 / 1999)  # Approximately 1.8 seconds

    # Close the cursor and connection when done
    cursor.close()
    conn.close()
    cursor_ticket.close()
    conn_ticket.close()

