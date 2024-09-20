import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
    
def preprocess_all_companies_data(df):
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

def retrieve_final_combined_data():
            # Connect to the database
        conn = sqlite3.connect('stock.db')

        # Retrieve the data from the combined table
        query = 'SELECT * FROM final_combined_data'  # Modify this query based on your actual table name
        data = pd.read_sql(query, conn)
        # Close the connection
        conn.close()

        stock_data = preprocess_all_companies_data(data)
        # Display the first few rows
        print(stock_data.columns)
        print(stock_data.shape)
        return stock_data

def divide_train_test(stock_data): 
    # Separate features and target
    X = stock_data.drop(columns=['current_price'])  # Replace 'target_column' with your actual target column name
    y = stock_data['current_price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert y_train to a tensor if it's not already one
    y_train = torch.tensor(y_train.values, dtype=torch.float64)  # Assuming y_train is a Pandas Series or similar
    # Convert y_test to a tensor if it's not already one
    y_test = torch.tensor(y_test.values, dtype=torch.float64)  # Assuming y_test is a Pandas Series or similar
    y_train = y_train.view(-1, 1)
    y_test = y_test.view(-1, 1)
    # Assuming X_train and X_test are your datasets (NumPy arrays or Pandas DataFrames)
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)

    # Custom scaling to avoid large values (this is optional based on your data)
    X_train = np.clip(X_train, -1e10, 1e10)
    X_test = np.clip(X_test, -1e10, 1e10)

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float64)
    X_test = torch.tensor(X_test, dtype=torch.float64)

    return X_train, X_test, y_train, y_test

def train_and_save_mlp(num_epochs = 100, lr = 0.001, stock_data = retrieve_final_combined_data()):
    X_train, X_test, y_train, y_test = divide_train_test(stock_data)

    input_size = X_train.shape[1]
    hidden_size = 128
    output_size = 1  # Assuming a regression task
    model = MLP(input_size, hidden_size, output_size)

    # Convert model to the same dtype as your data
    model = model.double()

    criterion = nn.MSELoss()  # Assuming regression
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    mlp_pth = 'mlp_model.pth'
    torch.save(model.state_dict(), mlp_pth)
    # Assuming model, X_test, y_test, and criterion are already defined

    # Ensure y_test is correctly shaped
    y_test = y_test.view(-1, 1)

    # Set the model to evaluation mode
    model.eval()

    # Perform inference on the test set
    with torch.no_grad():
        predictions = model(X_test)

    # Calculate the loss on the test set
    test_loss = criterion(predictions, y_test)
    print(f'Test Loss: {test_loss.item()}')

    # Optionally, calculate other metrics
    from sklearn.metrics import mean_absolute_error, r2_score

    # Convert predictions and y_test to numpy arrays for sklearn metrics
    predictions_np = predictions.numpy()
    y_test_np = y_test.numpy()

    # Calculate additional metrics
    mae = mean_absolute_error(y_test_np, predictions_np)
    r2 = r2_score(y_test_np, predictions_np)

    print(f'Mean Absolute Error: {mae}')
    print(f'R² Score: {r2}')

    return input_size, hidden_size, output_size, mlp_pth

def get_predictions(X, input_size, hidden_size, output_size, mlp_pth):
    model = MLP(input_size, hidden_size, output_size)
    model = model.double()
    model.load_state_dict(torch.load(mlp_pth))
    model.eval()
    with torch.no_grad():
        predictions = model(X)
        print(predictions)
        return predictions

def convert_tensor_compatibility(stock_data): 
    X = stock_data.drop(columns=['current_price'])  # Replace 'target_column' with your actual target column name

    X = X.astype(np.float64)

    # Custom scaling to avoid large values (this is optional based on your data)
    X = np.clip(X, -1e10, 1e10)

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float64)
    return X
