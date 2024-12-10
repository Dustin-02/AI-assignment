import pandas as pd
import numpy as np

def generate_synthetic_test_data(num_samples=200):
    np.random.seed(42)  # For reproducibility
    time_index = pd.date_range(start='2020-01-01', periods=num_samples, freq='D')
    
    # Generate random data for features
    volume = np.random.randint(100, 1000, size=num_samples)
    closing_price = np.random.uniform(10, 100, size=num_samples)
    sma = pd.Series(closing_price).rolling(window=5).mean().fillna(method='bfill')  # Simple Moving Average
    ema = pd.Series(closing_price).ewm(span=5, adjust=False).mean()  # Exponential Moving Average
    macd = ema - pd.Series(closing_price).rolling(window=12).mean().fillna(method='bfill')  # MACD

    # Create a DataFrame
    data = pd.DataFrame({
        'machine_record_index': range(num_samples),
        'Volume': volume,
        'Closing Price': closing_price,
        'SMA': sma,
        'EMA': ema,
        'MACD': macd
    })
    
    # Add a target variable for trends (1 for up, 0 for down)
    data['Trend'] = np.where(data['Closing Price'].shift(-1) > data['Closing Price'], 1, 0)  # 1 for up, 0 for down
    
    return data

test_data = generate_synthetic_test_data()
test_data.to_csv('synthetic_machine_data_TEST.csv', index=False)  # Save to CSV
print("Synthetic test data generated and saved to 'synthetic_test_data.csv'.")