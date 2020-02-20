deeplearning.ai - TensorFlow in Practice - Sequences, Time Series and Prediction

## Week 1 - Sequences and Prediction

### What is a time series?

**A**: An ordered sequences of values, usually equally spaced over time.

Time series are everywhere:

* Stock prices
* Weather forecasts
* Historical trends e.g. Moore's Law



### Univariate / Multivariate

* Univariate - single value at each time-step
* Multi-variate - multiple values at each time-step

Multi-variate can be useful for understanding relationship between data variables.

e.g. Global CO2 concentration vs global temp, death vs birth rates

Movement of a body can be plotted as a series of uni-variates, or a combined multivariate time series dataset. E.g. path of a car as it travels, latitute vs longitude (we can plot these a uni-variates) or with acceleration changing spaces between time-steps as multivariate.



### Machine Learning applied to time series

* Forecasting data
* Imputing - projecting back into the past before data was collected, or filling holes
* Anomaly detection - e.g. website logs, DOS attacks
* Patterns - e.g. soundwaves to detect words for speech detection



### Common patterns in time series

* **Trend**
* **Seasonality** - repeating patterns at predicted intervals, e.g. work production weeks vs weekdays, shopping trends, sports vieweing
* **Combined features** - increasing trend with seasonal peaks and troughs
* **Noise** - no trends or seasonality, random in nature and therefore not much can be done to predict it
* **Autocorrelation** - correlating with a delayed copy of itself called a lag. Time series has memory, dependent on sequences time series. Spikes call innovations i.e. cannot be predicted based on past values.
* **Multi-autocorrelated** - e.g. Impulse exponential delays with small oscillations in between
* Many time-series will have a **combination** of all of the above.
* **Non-Stationary Time Series** - e.g. trend in time, then big event and change in trend. E.g. financial data, with sudden disruptive tech breakthough, scandal and trend down after.

### Training

* If data is **Stationary** then the more data the better
* If data in **Non-Stationary** then you may be better off training on a specific/latest section that you want to predict from, e.g. downward trend after financial crash, but it depends on the data.

See the first Colab Notebook - Introduction to Time Series.



### Train, validation and test sets

#### Naive Forecasting

We could take the last value and use it as the prediction for the next value (i.e. the naive forecast lags one time-step being the actual data with the same value) - this is called Naive forecasting and can work surprisingly well to give a baseline performance/error to improve upon.

#### How to measure performance?

* We want to ensure that the train/valid/test sets have an entire season if the data is seasonal

**Method 1  - Fixed Partitioning - Train/Valid/Test split** 

* We then train on the **Training Period** and validate on the **Validation Period**. Then we can test the predictions on the **Test Period**. 

* If the predictions are good, then we can take an unusual step (in the case of previously seen methods) of retraining on the entire dataset **Train+Valid+Test** to give the model the most data and the **most recent** data available.

  

  ![train-valid-test-1](C:\Users\mikef\Documents\GitHub\dlai-tf-in-practice\course-4-sequences, time series and prediction\train-valid-test-1.png)



**Method 2  - Fixed Partitioning -  Train/Valid** 

* It is also common in practice to forego the Test Set altogether and to only use a **Train Period** and a **Valid Period** up to latest point in time:

  

![image-20200219100855256](C:\Users\mikef\AppData\Roaming\Typora\typora-user-images\image-20200219100855256.png)



**Method 3 - Roll-Forward Partitioning**:

* Start with a short **Training Period** and gradually increase it, e.g. by one day or one week at a time.
* At each iteration we train the model on the **Training Period** and we use it to forecast e.g. the following day, or following week in the **Validation period.**
* This could be seen as doing *Fixed-Partitioning* a number of times and continually refining the model as such.

![roll-forward-partitioning](C:\Users\mikef\Documents\GitHub\dlai-tf-in-practice\course-4-sequences, time series and prediction\roll-forward-partitioning.png)



In the course the focus is on *Fixed-Partitioning*, but we'll see code that allows us to implement *Roll-Forward Partitioning*.



### Metric for evaluating performance

* `errors = forecasts - actual`
* `mse = np.square(errors).mean()` - *Mean Squared Error* - squared errors to ensure only positive values.
* `rmse = np.sqrt(mse)` - *Root Mean Squared Error* - If we want the mean of our error calculation to be the same scale as the original errors, we use RMSE.
* `mae = np.abs(errors).mean()` - *Mean Absolute Error / Mean Absolute Deviation* - instead of squaring to get rid of negative values, it uses the absolute value. This doesn't penalise large errors as much as MSE does.
* `mape = np.abs(errors / x_valid).mean()` -  *Mean Absolute Percentage Error* - the mean ratio between the absolute error and the absolute value. This gives an idea of the size of the errors compared to the absolute values.

**Note:** If large errors are potentially dangerous and would be costly, then you may prefer to use the **MSE**. However, if your gain or your loss is just proportional to the size of the error then the **MAE** may be better.



With the Naive Forecast example we can use MAE using the following keras code:

`keras.metrics.mean_absolute_erro(x_valid, naive_forecast).numpy()`



### Moving average and differencing

A common and very simple forecasting method is a moving average. The plot would be the average value of the dataset over a fixed window e.g. 30 days as shown below:

![image-20200219100706265](C:\Users\mikef\AppData\Roaming\Typora\typora-user-images\image-20200219100706265.png)



* This eliminates a lot of the noise and gives a curve roughly emulating the original series, **but** does not anticipate seasonal variation or trend

* It can actually end up being worse than a Naive Forecast, depending on the forecasting window and the future events.

* This can be improved by removing seasonality and trend using a technique known as **Differencing**.

#### Differencing

  * Instead of studying the time series, we study the difference between the value at time **t** and the value at an earlier period (e.g. **365** days ago).

  * We get a difference time series (**t-365**), and can use a moving average to forecast this differenced time-series:

    

![image-20200219100751127](C:\Users\mikef\AppData\Roaming\Typora\typora-user-images\image-20200219100751127.png)



* To get the **final forecast** for the **original** time series, we need to add back the value at time **t-365**:

  

![image-20200219093748105](C:\Users\mikef\AppData\Roaming\Typora\typora-user-images\image-20200219093748105.png)



This gives quite a good forecast (better than Naive Forecasting), but can be quite noisy. The noise comes from the past values that were added back into the data.

* We can remove this noise by doing a moving average of the past data too:

  

![image-20200219094015814](C:\Users\mikef\AppData\Roaming\Typora\typora-user-images\image-20200219094015814.png)



This gives us a much better MSE than all the other approaches in this case. Simple approaches can sometimes work just fine, so be sure to check these before rushing into using deep learning!



### Trailing versus centered windows

In some cases using centred windows can be more accurate than trailing windows:

* Using a trailing window when computing the moving average of present values from **t-30days** to **t-1day**  ***vs.***
* using a centred window to compute the moving average of past values from one year ago, e.g. **t-1y-5days** to **t-1y+5days**. 

This is because we can't use centred windows to smooth *present* values, since we don't know *future* values.

However, to smooth *past* values, we can use centred windows.



### Forecasting

See the Google Colab notebook:

`TensorFlow in Practice - Course 4 S+P - Week 1 - Lesson 3 - Forecasting.ipynb`

This uses the statistical methods learned in this week to forecast the data. We will then compare this to the Machine Learning methods next week.



