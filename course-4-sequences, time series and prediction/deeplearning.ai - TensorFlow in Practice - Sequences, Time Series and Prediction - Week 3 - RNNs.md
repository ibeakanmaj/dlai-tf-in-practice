# deeplearning.ai - TensorFlow in Practice - Sequences, Time Series and Prediction

## Week 3 - Recurrent Neural Networks for Time Series

### Conceptual overview

* In this course we'll use an RNN that contains 2 Recurrent layers and 1 dense layer.

* With an RNN we can feed in batches of sequences and it will output a batch of forecasts.

* The Input X will be the windowed data with shape = [batch_size, # time-steps, # dimensionality of input at each time step]

  * dimensionality of input at each time step:

    * if a univariate time-series = 1

    * multivariate time-series = 2

      

* The RNN Layers are defined as follows:

![image-20200221125307849](C:\Users\mikef\AppData\Roaming\Typora\typora-user-images\image-20200221125307849.png)



* 1 RNN cell is repeated over and again per layer:

  * At each time-step, the memory cell takes the input value for that step and the cell-state input from the previous calculation:  i.e. inputs = X0 at time 0,  0 state input. It then calcualtes the output for that step, i.e. Y0 and the state vector input for the next step H0.
  * H0 and X1 fed into memory cell again => calculates Y1 etc, etc.
  * Repeat until we reach the end of our input dimension which is this case has 30 values.

  

### Shape of the RNN inputs and outputs in theory

![image-20200221132101350](C:\Users\mikef\AppData\Roaming\Typora\typora-user-images\image-20200221132101350.png)

#### Inputs

* The inputs are 3 dimensional, e.g. a univariate time series (e.g. # dimensionality of input at each time step  = **1**) with window of **30 ** time-steps and batch size of **4**:
  * input shape = [**4**, **30**, **1**]
* At each time step the memory cell input will be of dimensions *[batch_size, #dimensionality of input at each time step]*:
  * memory cell input shape = [**4**, **1**] matrix in this example as univariate

#### Outputs

* The cell will also take the input of the state matrix from the previous step. For the 1st timestep X1 this will be 0 and for the subsequent timesteps this will be the output of the previous cell (H#).

* Other than the state-vector/matrix, the cell will output a Y value - the prediction, with dimensions:

  * **Ystep** = ***[batch_size, # of units]***
  * So, if the memory cell is comprised of **3** neurons and the batch size is **4**, then the output matrix will be:
    * [**4**, **3**]

* So the full output of the layer, Y,  is 3 dimensional with dimensions:

  * **Y** = ***[batch_size, # of units, # overall steps]***
  * So in this example with **30** steps:
    * [**4**, **3**, **30**]

* In a simple RNN the state output H# is just a copy of the output matrix Y#. e.g. H0 = Y0, H1 = Y1 etc. So at each timestep the memory cell gets as inputs both the current input and the previous output.

  

### Sequence to Vector

In some cases you might want to input a sequence, but not an output at each time step and only a single output per batch.

In reality all you do is ignore all the outputs but the last one.

**When using Keras in TensorFlow this is the default behaviour (only a single output per batch), so if you want a recurrent layer to output at each timestep you need to specify:**

```python
return_sequences = True
```

in the LSTM, GRU or other RNN layer parameters. This will need to be done when stacking on RNN layer on top of another.



![image-20200221132225936](C:\Users\mikef\AppData\Roaming\Typora\typora-user-images\image-20200221132225936.png)



### Input sizes and outputting a sequence using tf.keras

If we consider an RNN with 2 recurrent layers -  a **Sequence-to-Vector RNN**: 

![image-20200221143150999](C:\Users\mikef\AppData\Roaming\Typora\typora-user-images\image-20200221143150999.png)



* the first layer has the parameter: `return_sequences = True` so that it will output to the next layer which does not have this parameter setting, so the second layer only outputs the final step (as default).

* the `input_shape = [None, 1]` - TensorFlow assumes the first dimension is the batch size and that it can have any dimension at all, so that you don't need to define it and it is not shown. The next dimension is the number of timesteps which we can set to `None` so that the RNN can handle sequences of any length. The last dimension is just `1` because we're using a univariate time-series.

* If we set `return_sequences = True` in all the recurrent layers then they will all output sequences and the dense layer will get a sequence as its input. Keras handles this by using the same Dense layer at each timestep. This gives a **Sequence-to-Sequence RNN**:

  * It's fed a batch of sequences and returns a batch of the same length
  *  The dimensionality may not always match between input and output, it depends on the amount of units in the memory cell

  ![image-20200221144059403](C:\Users\mikef\AppData\Roaming\Typora\typora-user-images\image-20200221144059403.png)





