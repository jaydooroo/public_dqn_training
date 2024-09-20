import pandas as pd
import requests
import sqlite3
import time
from datetime import datetime
import yfinance as yf
import sqlite3
from datetime import datetime
# Function to fetch data from Alpha Vantage
def get_alpha_vantage_data(api_key, function, **kwargs):
    url = f"https://www.alphavantage.co/query?function={function}&apikey={api_key}"
    for key, value in kwargs.items():
        url += f"&{key}={value}"
    response = requests.get(url)
    data = response.json()
    if 'data' in data:
        df = pd.DataFrame(data['data'])
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df.set_index('date', inplace=True)

        # Clean the data: Convert numeric values to floats, and replace non-numeric values with NaN
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # Drop rows with missing or non-numeric values
        df.dropna(inplace=True)

        df.columns = [function]
        return df
    else:
        print(f"Error fetching data for {function}: {data.get('Note', 'No additional information available.')}")
        return pd.DataFrame()

# Function to fetch all economic data
def fetch_all_economic_data(api_key = "D71CLYCNNCG0DAIC"):
    economic_data = pd.DataFrame()

    indicators = [
        ('CPI', {}),
        ('UNEMPLOYMENT', {}),
        ('FEDERAL_FUNDS_RATE', {}),
        ('RETAIL_SALES', {}),
    ]

    for function, params in indicators:
        data = get_alpha_vantage_data(api_key, function, **params)
        if not data.empty:
            if economic_data.empty:
                economic_data = data
            else:
                economic_data = economic_data.merge(data, left_index=True, right_index=True, how='outer')
        # Sleep for 12 seconds to stay within the 5 requests per minute limit
        time.sleep(13)

    return economic_data

# Function to create the table if it doesn't exist
def create_table_if_not_exists_econ(conn, table_name='economic_history_data'):
    cursor = conn.cursor()
    create_table_query = f'''
    CREATE TABLE IF NOT EXISTS {table_name} (
        date TEXT PRIMARY KEY,
        CPI REAL NULL,
        UNEMPLOYMENT REAL NULL,
        FEDERAL_FUNDS_RATE REAL NULL,
        RETAIL_SALES REAL NULL
    );
    '''
    cursor.execute(create_table_query)
    conn.commit()

def drop_table(db_name='stock.db', table_name='economic_history_data'):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.commit()
    conn.close()

    print(f"Table '{table_name}' has been deleted.")

# Function to save DataFrame to SQLite database by directly inserting it
def save_to_sqlite_econ(df, db_name='stock.db', table_name='economic_history_data'):
    conn = sqlite3.connect(db_name)
    # Create table if it doesn't exist
    create_table_if_not_exists_econ(conn, table_name)

    # Directly insert data into the database
    df.to_sql(table_name, conn, if_exists='append', index=True)
    print(f"Data inserted into {db_name} in the table {table_name}.")

    conn.close()
        
# Function to get FRED data
def get_fred_data(api_key, series_id, start_date, end_date):
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date,
        'observation_end': end_date,
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data['observations'])
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df.set_index('date', inplace=True)
    df = df[['value']]
    df.columns = [series_id]
    return df

def get_latest_fred_data(api_key, series_id):
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'sort_order': 'desc',  # Sort by descending date to get the latest first
        'limit': 1,  # Only get the latest data point
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data['observations'])
    print(df)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df.set_index('date', inplace=True)
    df = df[['value']]
    df.columns = [series_id]
    return df

# Function to fetch multiple treasury yields
def fetch_treasury_yields(api_key, start_date, end_date):
    series_ids = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']
    treasury_data = pd.DataFrame()

    for series_id in series_ids:
        data = get_fred_data(api_key, series_id, start_date, end_date)
        if treasury_data.empty:
            treasury_data = data
        else:
            treasury_data = treasury_data.join(data, how='outer')

    return treasury_data

# Function to create the table if it doesn't exist
def create_table_if_not_exists_treasury_yields(conn):
    cursor = conn.cursor()
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS treasury_yields (
        date TEXT PRIMARY KEY,
        DGS1MO REAL NULL,
        DGS3MO REAL NULL,
        DGS6MO REAL NULL,
        DGS1 REAL NULL,
        DGS2 REAL NULL,
        DGS3 REAL NULL,
        DGS5 REAL NULL,
        DGS7 REAL NULL,
        DGS10 REAL NULL,
        DGS20 REAL NULL,
        DGS30 REAL NULL
    );
    '''
    cursor.execute(create_table_query)
    conn.commit()

# Function to save DataFrame to SQLite database
def save_to_sqlite_treasury(df, db_name='stock.db', table_name='treasury_yields'):
    conn = sqlite3.connect(db_name)
    # Create table if it doesn't exist
    create_table_if_not_exists_treasury_yields(conn)

    # Directly insert data into the database
    df.to_sql(table_name, conn, if_exists='append', index=True)
    print(f"Data inserted into {db_name} in the table {table_name}.")

    conn.close()

def save_latest_to_sqlite(df, db_name='stock.db', table_name='treasury_yields'):
    conn = sqlite3.connect(db_name)
    create_table_if_not_exists_treasury_yields(conn)

    # Check if the new data is already in the database
    existing_data = pd.read_sql(f"SELECT * FROM {table_name} WHERE date = ?", conn, params=(df.index[0].strftime('%Y-%m-%d'),))
    
    if existing_data.empty:
        # Insert the new data if it's not already there
        df.to_sql(table_name, conn, if_exists='append', index=True)
        print(f"New data inserted for {df.index[0].strftime('%Y-%m-%d')}")
    else:
        print(f"Data for {df.index[0].strftime('%Y-%m-%d')} is already up to date.")

    conn.close()

# for live update for the treasury data. 
def fetch_latest_and_store_data(api_key):
    try:
        # Get the latest data
        series_ids = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']
        treasury_data = pd.DataFrame()

        for series_id in series_ids:
            data = get_latest_fred_data(api_key, series_id)
            if treasury_data.empty:
                treasury_data = data
            else:
                treasury_data = treasury_data.join(data, how='outer')

        # Save to the SQLite database
        save_latest_to_sqlite(treasury_data)
    except Exception as e:
        print(f"Error occurred: {e}")




def add_dxy_to_treasury(): 
    # Get today's date
    today = datetime.today().strftime('%Y-%m-%d')

    # Fetch U.S. Dollar Index data
    dxy = yf.download("DX-Y.NYB", start="2023-01-01", end=today)

    # Print the most recent data points (for debugging/verification)
    print(dxy.tail())

    # Connect to SQLite database (replace 'your_database.db' with your actual database name)
    conn = sqlite3.connect('stock.db')
    cursor = conn.cursor()

    # Ensure the table exists (modify the create statement according to your schema)
    create_table_query = '''
        CREATE TABLE IF NOT EXISTS treasury_yields (
            date TEXT PRIMARY KEY,
            DGS1MO REAL NULL,
            DGS3MO REAL NULL,
            DGS6MO REAL NULL,
            DGS1 REAL NULL,
            DGS2 REAL NULL,
            DGS3 REAL NULL,
            DGS5 REAL NULL,
            DGS7 REAL NULL,
            DGS10 REAL NULL,
            DGS20 REAL NULL,
            DGS30 REAL NULL,
            DXY REAL NULL
        );
        '''
    cursor.execute(create_table_query)
    conn.commit()

    try:
        cursor.execute("ALTER TABLE treasury_yields ADD COLUMN DXY REAL NULL;")
        conn.commit()
        print("DXY column added to the treasury_yields table.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("DXY column already exists.")
        else:
            raise e

    # Insert DXY data into the table
    for index, row in dxy.iterrows():
        date = index.strftime('%Y-%m-%d')
        dxy_value = row['Close']  # Use the 'Close' price as the value for DXY

        # Check if the date already exists
        cursor.execute('SELECT date FROM treasury_yields WHERE date = ?', (date,))
        result = cursor.fetchone()

        if result:
            # If the date exists, update the DXY value
            cursor.execute('''
                UPDATE treasury_yields
                SET DXY = ?
                WHERE date = ?
            ''', (dxy_value, date))
        else:
            # If the date does not exist, insert a new row
            cursor.execute('''
                INSERT INTO treasury_yields (date, DXY)
                VALUES (?, ?)
            ''', (date, dxy_value))

        conn.commit()

    # Close the database connection
    cursor.close()
    conn.close()

    print("Dollar Index data has been successfully inserted/updated in the treasury_yields table.")


def combine_economic_treasury_data(): 
# Connect to your database
    conn = sqlite3.connect('stock.db')

    # Combine the data using the SQL query
    query = '''
    SELECT 
        t.date, 
        t.DGS1MO, t.DGS3MO, t.DGS6MO, t.DGS1, t.DGS2, t.DGS3, t.DGS5, t.DGS7, t.DGS10, t.DGS20, t.DGS30, t.DXY,
        e.CPI, e.UNEMPLOYMENT, e.FEDERAL_FUNDS_RATE, e.RETAIL_SALES
    FROM 
        treasury_yields t
    LEFT JOIN 
        economic_history_data e 
    ON 
        e.date = (SELECT max(e2.date) 
                FROM economic_history_data e2 
                WHERE e2.date <= t.date)
    '''

    # Execute the query and fetch the result into a DataFrame
    combined_data = pd.read_sql_query(query, conn)

    # If you want to save this combined data into a new table
    combined_data.to_sql('combined_economic_treasury_data', conn, if_exists='replace', index=False)

    # Close the connection
    conn.close()

    # Display the first few rows of the combined data
    print(combined_data.head())


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

def create_latest_stock_data(): 
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
        sector TEXT,
        timestamp TEXT,
        retrieval_time TEXT
    );
    '''
    cursor.execute(create_table_query)
    conn.commit()