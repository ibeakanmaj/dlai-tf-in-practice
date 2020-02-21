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

#### More on single layer neural network



