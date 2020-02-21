# deeplearning.ai - TensorFlow in Practice - Sequences, Time Series and Prediction

## Week 2 - Deep Neural Networks for Time Series

### Preparing features and labels

* Feature/Input, X  - a number of values we are taking i.e. the window / sample data
* Label,  y - the next value in the series after the Input data

See the `Preparing Features and Labels` Colab notebook for the TensorFlow implementation of creating datasets, windowing, truncating,  shuffling and  batching using numpy, TF and lambda functions.



### Feeding windowed dataset into neural network

See the `Single Layer NN with Time Series` Colab notebook for in-depth comments of the process.

Import element:

See the `windowed_dataset()` function:

````python
# Takes in a data series, window size and batch size and shuffle and created windowed dataset

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series) # Create dataset
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True) # Slice the data up into windows, each one shifted by one time step, drop remainder to standardize size
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1)) # flatten the data to window size + 1
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1])) # shuffle the data - shuffle buffer speeds up - if you have 100,000 items and set to 1000, will fill with first 1000 elements, pick one at random, then replace that the 1001 element before randomly picking again and so on - so speeds things up by effectively choosing from smaller data sample
  # map lambda function splits the suffled data into the x (all but last) and y (last) data items
  dataset = dataset.batch(batch_size).prefetch(1) # The data is then batched and returned
  return dataset 
````

1. Takes in a data series, window size and batch size and shuffle
2. Creates a dataset
3. Slice the data up into windows, each one shifted by one time step, drop remainder to standardize size
4. Flatten the data to window size + 1
5. Shuffle the data - shuffle buffer speeds up the process - e.g. if you have 100,000 items and set to 1000, will fill with first 1000 elements, pick one at random, then replace that the 1001 element before randomly picking again and so on - so speeds things up by effectively choosing from smaller data sample
6. Map lambda function splits the shuffled data into the x (all but last) and y (last) data items
7. The data is then batched and returned



Also see the **prediction** - we are passing in windowed time series data (X) and using the weights (W) of the model to do standard linear regression for each window:

`y = W0 * X0 + .... + W19 * X19 + b`

#### 

## Week 2 Quiz

**LATEST SUBMISSION GRADE**

100%

1.Question 1

**What is a windowed dataset?**

***A fixed-size subset of a time series***

The time series aligned to a fixed shape

A consistent set of subsets of a time series

There’s no such thing



2.Question 2

**What does ‘drop_remainder=true’ do?**



It ensures that the data is all the same shape

***It ensures that all rows in the data window are the same length by cropping data***

It ensures that all rows in the data window are the same length by adding data

It ensures that all data is used



3.Question 3

**What’s the correct line of code to split an n column window into n-1 columns for features and 1 column for a label**



dataset = dataset.map(lambda window: (window[n-1], window[1]))

***dataset = dataset.map(lambda window: (window[:-1], window[-1:]))***

dataset = dataset.map(lambda window: (window[-1:], window[:-1]))

dataset = dataset.map(lambda window: (window[n], window[1]))



4.Question 4

**What does MSE stand for?**



***Mean Squared error***

Mean Second error

Mean Slight error

Mean Series error





5.Question 5

**What does MAE stand for?**



Mean Average Error

Mean Advanced Error

***Mean Absolute Error***

Mean Active Error





6.Question 6

**If time values are in time[], series values are in series[] and we want to split the series into training and validation at time 1000, what is the correct code?**



``` Python
# Correct
time_train = time[:split_time] 
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
```

``` 
time_train = time[split_time]
x_train = series[split_time]
time_valid = time[split_time]
x_valid = series[split_time]
```

```
time_train = time[split_time]
x_train = series[split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
```

```
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time]
x_valid = series[split_time]
```





7.Question 7

**If you want to inspect the learned parameters in a layer after training, what’s a good technique to use?**



Run the model with unit data and inspect the output for that layer

Decompile the model and inspect the parameter set for that layer

Iterate through the layers dataset of the model to find the layer you want

***Assign a variable to the layer and add it to the model using that variable. Inspect its properties after training***





8.Question 8

**How do you set the learning rate of the SGD optimizer?** 



You can’t set it

Use the RateOfLearning property

***Use the lr property***

Use the Rate property 



9.Question 9

**If you want to amend the learning rate of the optimizer on the fly, after each epoch, what do you do?**



Use a LearningRateScheduler and pass it as a parameter to a callback

Callback to a custom function and change the SGD property

***Use a LearningRateScheduler object in the callbacks namespace and assign that to the callback***

You can’t set it



