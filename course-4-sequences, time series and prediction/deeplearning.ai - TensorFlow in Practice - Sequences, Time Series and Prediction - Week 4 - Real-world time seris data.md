# deeplearning.ai - TensorFlow in Practice - Sequences, Time Series and Prediction

# Week 4 - Real-world time series data

See the notebooks in Week 4.

## Convolutions

Now we combine convolutions with LSTMs to get a very nicely fitting model.

We will use a `Conv1D` layer at the beginning of the sequential stack before the LSTM layers, using the following parameter settings:

* `filters = 32` - using 32 filters
* `kernel_size = 5` - 5 number window and multiply the values in that window by the filter values, in much the same way as image convolutions are done.
* `stride=1, padding = causal, activation = 'relu', input_shape = [None,1]`



## Data Input

We've removed the Lambda layer that reshaped the input data for the LSTMs, so we must specify the input shape on the `Conv1D` layer instead. We do this via the the following, to expand the data dimension before we process it:

 ` series = tf.expand_dims(series, axis=-1)`



## Learning Rate finder

Also used is the previously mentioned learning rate finder method (see week 3 notes.



## Batch Sizing

RNNs can be very sensitive to batch sizing and you will need to tweak the sizing in order to optimise the model based on the dataset.

E.g. in the notebook `deeplearning.ai - TensorFlow in Practice - Course 4 S+P - Week 4 - Lesson 1.ipynb` we can see that there is a lot of noise and instability in the test results. 

One reason for this is **small batch-size** introducing random noise, so this can be thought of as a **hyperparameter** that needs to be tuned and is relevant to the dataset.

To learn more, see the Deep Learning course /videos.

https://www.youtube.com/watch?v=4qJaSmvhxi8



## Real Data Analysis - Sunspots

We will now apply what we've learned to real data - sunspot data taken on a monthly basis from 1749 until 2018. Sunspots have seasonal cycles approximately every 11 years, so we can try to predict this data.

See the notebook: `deeplearning.ai - TensorFlow in Practice - Course 4 S+P - Week 4 - Lesson 5 - Conv1D and LSTM Sunspot times series prediction.ipynb`



### Data

The dataset is from Kaggle:

http://www.kaggle.com/robervalt/sunspots

It's a .csv dataset with the:

*  first column being an index
* second the date YYYY-MM-DD
* the third being the date of that month the measurement was taken (??)
* an average monthly amount of sunspots at the end of the month

### Train and tune the model

Remember, one size / one method does not fit all when working with time series, particularly seasonal data.

In the pure DNN notebook the results are poor: MAE 19.

We can think of the windowing and slicing of the timeseries data as **hyperparameters** that need setting to get the best from the model.

The clue to the problem could be the `window size`:

* the initial `window size` is set to 20 from earlier, so the training window sizes are 20 time slices of data. Each time slice is a month in real time, so this gives a window of **2 years**.
* Our sunspots have a cycle of 11 / 22 years.  However, we probably don't need a huge window like this as it really is a shorter time series. if it's too large the MAE gets worse. 
* So let's try something like the original value

Let's look at `split_time` too:

* the initial setting is **1000**. The dataset has around 3500 items, so we are given the training set 1000 and the validation set 2500, which is clearly too much on the validation side. Let's change it to **3000**. This gives an MAE of 15.

Let's look at the optimizer  `learning_rate`:

* We can use the learning rate finder method of increasing the learning rate every epoch and seeing the loss, to find the correct learning rate.
* Remember to use `tf.keras.backend.clear_session()` to clear the results of this mock training run so that it doesn't affect the actual training run.



### Prediction

The `window_size` is 30 steps and dataset is 3235 steps long, so to predict the next value after the end of the data we use the following:

`model.predict(series[3205:3235][np.newaxis])`



