import pandas as pd
import statsmodels.api as sm
import datetime

# Load the data from the provided CSV file
data = pd.read_csv('Nat_Gas.csv')

# Convert the 'Dates' column to datetime format
data['Dates'] = pd.to_datetime(data['Dates'], format='%m/%d/%y')

# Set the 'Dates' column as the index
data.set_index('Dates', inplace=True)

# Sort the data by date
data.sort_index(inplace=True)

# Define a function to estimate gas prices for a given date


def estimate_gas_price(target_date):
    # Split the data into training and test sets
    train_data = data[data.index <= target_date]

    # Fit a time series model (e.g., SARIMA) to the training data
    # You may need to adjust the model parameters based on the data characteristics
    model = sm.tsa.SARIMAX(train_data['Prices'], order=(
        1, 1, 1), seasonal_order=(0, 1, 1, 12))
    results = model.fit()

    # Use the fitted model to forecast prices for the next 12 months
    forecast = results.get_forecast(steps=12)

    # Extract the forecasted prices for the next 12 months
    forecast_mean = forecast.predicted_mean
    forecast_index = pd.date_range(
        start=train_data.index[-1] + pd.DateOffset(months=1), periods=12, freq='M')

    # Return the estimated price for the target date and the forecasted prices for the next year
    return forecast_mean.loc[forecast_index]


# Example usage:
target_date = datetime.datetime(2023, 9, 15)
estimated_price = estimate_gas_price(target_date)
print(
    f"Estimated price for {target_date.strftime('%B %Y')}: {estimated_price.iloc[0]:.2f}")
print("Estimated prices for the next 12 months:")
print(estimated_price)


"""
import pandas as pd
import statsmodels.api as sm
import datetime

# Load the data from the provided CSV file
data = pd.read_csv('Nat_Gas.csv')

# Convert the 'Dates' column to datetime format
data['Dates'] = pd.to_datetime(data['Dates'], format='%m/%d/%y')

# Set the 'Dates' column as the index
data.set_index('Dates', inplace=True)

# Sort the data by date
data.sort_index(inplace=True)

# Define a function to estimate gas prices for a given date


def estimate_gas_price(target_date):
    # Split the data into training and test sets
    train_data = data[data.index <= target_date]

    # Fit a time series model (e.g., SARIMA) to the training data
    # You may need to adjust the model parameters based on the data characteristics
    model = sm.tsa.SARIMAX(train_data['Prices'], order=(
        1, 1, 1), seasonal_order=(0, 1, 1, 12))
    results = model.fit()

    # Use the fitted model to forecast prices for the next 12 months
    forecast = results.get_forecast(steps=12)

    # Create hypothetical future dates for forecasting
    future_dates = pd.date_range(
        start=train_data.index[-1], periods=13, freq='M')[1:]

    # Extract the forecasted prices for the next 12 months
    forecast_mean = forecast.predicted_mean
    forecast_mean.index = future_dates

    # Return the estimated price for the target date and the forecasted prices for the next year
    return forecast_mean.loc[forecast_mean.index >= target_date]


# Example usage:
target_date = datetime.datetime(2023, 3, 15)
estimated_price = estimate_gas_price(target_date)
print(
    f"Estimated price for {target_date.strftime('%B %Y')}: {estimated_price.iloc[0]:.2f}")
print("Estimated prices for the next 12 months:")
print(estimated_price)
"""
