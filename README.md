# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
~~~
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
data = pd.read_csv('Electric_Production.csv')
print(data.head())
production = data['IPG2211A2N'].dropna() 
ar1 = np.array([1, -0.5])  # AR(1) coefficient
ma1 = np.array([1, 0.5])   # MA(1) coefficient
arma11_process = ArmaProcess(ar1, ma1)
arma11_sample = arma11_process.generate_sample(nsample=len(production))
ar2 = np.array([1, -0.5, 0.25])  # AR(2) coefficients
ma2 = np.array([1, 0.4, 0.3])    # MA(2) coefficients
arma22_process = ArmaProcess(ar2, ma2)
arma22_sample = arma22_process.generate_sample(nsample=len(production))
plt.figure(figsize=(14, 8))
plt.subplot(221)
plot_acf(arma11_sample, lags=20, ax=plt.gca(), title='ACF of Simulated ARMA(1,1)')
plt.subplot(222)
plot_pacf(arma11_sample, lags=20, ax=plt.gca(), title='PACF of Simulated ARMA(1,1)')
plt.subplot(223)
plot_acf(arma22_sample, lags=20, ax=plt.gca(), title='ACF of Simulated ARMA(2,2)')
plt.subplot(224)
plot_pacf(arma22_sample, lags=20, ax=plt.gca(), title='PACF of Simulated ARMA(2,2)')
plt.tight_layout()
plt.show()
~~~

OUTPUT:
SIMULATED ARMA(1,1) PROCESS:


Partial Autocorrelation
<img width="394" alt="image" src="https://github.com/user-attachments/assets/b4b5800c-7a0f-4da3-bfcd-d4dc1e2428a8">


Autocorrelation
<img width="400" alt="image" src="https://github.com/user-attachments/assets/6abcf8d8-a564-489b-9680-19a23c147fb5">


SIMULATED ARMA(2,2) PROCESS:

Partial Autocorrelation


<img width="391" alt="image" src="https://github.com/user-attachments/assets/437a4ad0-f498-4c35-b164-a4566eedfe45">


Autocorrelation
<img width="394" alt="image" src="https://github.com/user-attachments/assets/adf99b01-87e3-48d6-828f-e800abc8c204">
RESULT:
Thus, a python program is created for ARMA Model successfully.
