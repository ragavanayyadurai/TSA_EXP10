# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the Summer Olympics Medals data
data = pd.read_csv('/content/Summer_olympic_Medals.csv')

# Create a 'Total_Medals' column by summing Gold, Silver, and Bronze columns
data['Total_Medals'] = data['Gold'] + data['Silver'] + data['Bronze']

# Convert 'Year' to datetime format
data['Year'] = pd.to_datetime(data['Year'], format='%Y')

# Aggregate medals by year
yearly_medals = data.groupby(data['Year'].dt.year)['Total_Medals'].sum().reset_index()
yearly_medals.columns = ['Year', 'Total_Medals']

# Set 'Year' as the index
yearly_medals['Year'] = pd.to_datetime(yearly_medals['Year'], format='%Y')
yearly_medals.set_index('Year', inplace=True)

# Plot the yearly total medals
plt.plot(yearly_medals.index, yearly_medals['Total_Medals'])
plt.xlabel('Year')
plt.ylabel('Total Medals')
plt.title('Total Medals Time Series')
plt.show()

# Check stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

check_stationarity(yearly_medals['Total_Medals'])

# Plot ACF and PACF
plot_acf(yearly_medals['Total_Medals'])
plt.show()
plot_pacf(yearly_medals['Total_Medals'])
plt.show()

# Split the data into training and testing sets
train_size = int(len(yearly_medals) * 0.8)
train, test = yearly_medals['Total_Medals'][:train_size], yearly_medals['Total_Medals'][train_size:]

# Fit the SARIMA model (adjusted for 4-year seasonality)
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
sarima_result = sarima_model.fit()

# Generate predictions
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot actual vs predicted total medals
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Year')
plt.ylabel('Total Medals')
plt.title('SARIMA Model Predictions for Total Olympic Medals')
plt.legend()
plt.xticks(rotation=45)
plt.show()
```

### OUTPUT:
![image](https://github.com/user-attachments/assets/8b1d6945-e89a-44c3-889c-08ed3b026278)
![image](https://github.com/user-attachments/assets/6f5f9111-c772-4dd8-89ad-116640294138)
![image](https://github.com/user-attachments/assets/9210bfb1-120c-46ea-91d7-cd11ec16098e)
![image](https://github.com/user-attachments/assets/8b01d5d3-48c9-4b03-925c-5013d9c48b94)


### RESULT:
Thus the program run successfully based on the SARIMA model.
