# Ex.No: 6 HOLT WINTERS METHOD
## DATE: 30.09.2025
### AIM:
To implement the Holt Winters Method Model using Python.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column asdatetime, set it as index, and perform some initial data exploration
3. Resample it to a monthly frequency beginning of the month
4. You plot the time series data, and determine whether it has additive/multiplicativetrend/seasonality
5. Split test,train data,create a model using Holt-Winters method, train with train data and
6.  Evaluate the model predictions against test data
7. Create teh final model and predict future data and plot it

### PROGRAM:

```
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Load the dataset,perform data exploration
data = pd.read_csv('/content/AirPassengers.csv', parse_dates=['Month'],index_col='Month')

data_monthly = data.resample('MS').sum()   #Month start

data_monthly.head()
data_monthly.plot()

# Scale the data and check for seasonality
scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(), index=data_monthly.index)
scaled_data.plot() # The data seems to have additive trend and multiplicative seasonality

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data_monthly, model="additive")
decomposition.plot()
plt.show()

scaled_data=scaled_data+1 # multiplicative seasonality cant handle non postive values, ye
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul').fit()
test_predictions_add = model_add.forecast(steps=len(test_data))
ax=train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add","test_data"])
ax.set_title('Visual evaluation')
np.sqrt(mean_squared_error(test_data, test_predictions_add))
np.sqrt(scaled_data.var()),scaled_data.mean()

final_model = ExponentialSmoothing(data_monthly, trend='add', seasonal='mul', seasonal_periods=12).fit()
final_predictions = final_model.forecast(steps=int(len(data_monthly)/4)) #for next year
ax=data_monthly.plot()
final_predictions.plot(ax=ax)
ax.legend(["data_monthly", "final_predictions"])
ax.set_xlabel('Number of monthly passengers')
ax.set_ylabel('Months')
ax.set_title('Prediction')


```

# OUTPUT:
##### Scaled_data plot:
<img width="598" height="402" alt="image" src="https://github.com/user-attachments/assets/62725e26-4202-47fd-81e6-48533504bf5f" />

#### Decomposed plot:

<img width="686" height="452" alt="image" src="https://github.com/user-attachments/assets/6bb527fa-5e9a-4e8f-b04e-97d8165ca554" />

#####  Test prediction:
<img width="604" height="459" alt="image" src="https://github.com/user-attachments/assets/8060c8df-6f26-4275-8adf-e1f9a8337dc9" />

<img width="646" height="450" alt="image" src="https://github.com/user-attachments/assets/fd96fda9-8978-4ebc-9275-6c1945f73e28" />


## RESULT:
Thus the program run successfully based on the Holt Winters Method model.
