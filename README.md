# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 19/05/2026

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def arima_model(data, target_variable, order):
    # Ensure data index is datetime for ARIMA
    if not isinstance(data.index, pd.DatetimeIndex):
        print("Warning: Data index is not DatetimeIndex. Attempting to convert.")
        # If 'Date' column exists, try to set it as index
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            data.sort_index(inplace=True)
        else:
            # If no 'Date' column, this model won't work without a time index.
            raise TypeError("Data must have a DatetimeIndex or a 'Date' column for ARIMA modeling.")

    # Set frequency for the DatetimeIndex to avoid ValueWarning from statsmodels
    # Assuming monthly data, 'MS' for Month Start frequency
    data = data.asfreq('MS')

    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size].copy(), data[train_size:].copy()

    # Handle potential non-numeric data in target variable using .loc to avoid SettingWithCopyWarning
    train_data.loc[:, target_variable] = pd.to_numeric(train_data[target_variable], errors='coerce')
    test_data.loc[:, target_variable] = pd.to_numeric(test_data[target_variable], errors='coerce')

    # Drop NaNs that resulted from coercion, as ARIMA cannot handle them
    train_data = train_data.dropna(subset=[target_variable])
    test_data = test_data.dropna(subset=[target_variable])

    if train_data.empty or test_data.empty:
        print("Error: Training or testing data is empty after handling non-numeric values or NaNs.")
        return

    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()

    forecast_result = fitted_model.get_forecast(steps=len(test_data))
    forecast = forecast_result.predicted_mean

    # Ensure forecast index aligns with test_data for RMSE calculation and plotting
    forecast.index = test_data.index

    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))

    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data')
    plt.plot(test_data.index, forecast, label='Forecasted Data')
    plt.xlabel('Date')
    plt.ylabel(target_variable)
    plt.title('ARIMA Forecasting for ' + target_variable)
    plt.legend()
    plt.show()

    print("Root Mean Squared Error (RMSE):", rmse)

# --- Data Preprocessing ---
# Create a 'Date' column from 'Year' and 'Month'
data['Date'] = pd.to_datetime(data['Year'].astype(str) + '-' + data['Month'].astype(str) + '-01')
# Set 'Date' as the index for time series analysis
data.set_index('Date', inplace=True)
data.sort_index(inplace=True) # Ensure chronological order

# Call the function with the appropriate target variable
# Using 'Sales_Amount' as the target variable
arima_model(data, 'Sales_Amount', order=(5,1,5))

```

### OUTPUT:
<img width="1041" height="611" alt="image" src="https://github.com/user-attachments/assets/c75e58ec-aeab-4baf-b747-34ea3b17ff8f" />



### RESULT:
Thus the program run successfully based on the ARIMA model using python.
