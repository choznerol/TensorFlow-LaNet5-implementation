
Convolutional Neuron Network LeNet-5 implemented by Google TensorFlow
=============

------------

<a src="http://yann.lecun.com/exdb/lenet/index.html">
LeNet-5</a> is a classic CNN model which achieves 0.95% error rate on MNIST.

<img src="http://yann.lecun.com/exdb/lenet/gifs/asamples.gif">

This TensorFlow implementation is modified from a TensorFlow <a src="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/4_convolutions.ipynb">example</a>. The final test accuracy is 90.4% (further turing should be able to improve this result)

## Step 1: Import modules and load data


```python
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
from IPython.display import display, HTML
```


```python
pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
```

    Training set (200000, 28, 28) (200000,)
    Validation set (10000, 28, 28) (10000,)
    Test set (10000, 28, 28) (10000,)


## Step 2: Reformat into a TensorFlow-friendly shape:
- convolutions need the image data formatted as a cube (width by height by #channels)
- labels as float 1-hot encodings.


```python
image_size = 28
num_labels = 10
num_channels = 1 # grayscale

import numpy as np

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
```

    Training set (200000, 28, 28, 1) (200000, 10)
    Validation set (10000, 28, 28, 1) (10000, 10)
    Test set (10000, 28, 28, 1) (10000, 10)



```python
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
```

## Step3: Define a data flow graph to representing a TensorFlow computation

The model structure of LeNet-5:
<img src="https://www.researchgate.net/profile/Haohan_Wang/publication/282997080/figure/fig10/AS:305939199610894@1449952997905/Figure-10-Architecture-of-LeNet-5-one-of-the-first-initial-architectures-of-CNN.png">


```python
batch_size = 16
patch_size = 5
C1_depth = 6
C3_depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset) ## the shape is (10000, 28, 28)
  tf_test_dataset = tf.constant(test_dataset)   ## the shape is (10000, 28, 28)
  
  # Variables.
  C1_filter = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, C1_depth], stddev=0.1))
  C1_biases = tf.Variable(tf.zeros([C1_depth]))
  C3_filter = tf.Variable(tf.truncated_normal([patch_size, patch_size, C1_depth, C3_depth], stddev=0.1))
  C3_biases = tf.Variable(tf.constant(1.0, shape=[C3_depth]))
  C5_weights = tf.Variable(tf.truncated_normal([400, 120], stddev=0.1))
  C5_biases = tf.Variable(tf.constant(1.0, shape=[120]))
  F6_weights = tf.Variable(tf.truncated_normal([120, 84], stddev=0.1))
  F6_biases = tf.Variable(tf.constant(1.0, shape=[84]))
  OUTPUT_weights = tf.Variable(tf.truncated_normal([84, num_labels], stddev=0.1))
  OUTPUT_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
    
  # Model.
    ## LeNet-5 : 
    ## 1@32x32  --conv 5x5-->  6@28x28  --max pool 2x2-->  6@14x14  --conv 5x5-->  16@10x10  --max pool 2x2--> 16@5x5
    ## --full cont--> 120 --full cont--> 84 --full cont--> 10
  def model(data):
    print('         [batch, height, width, channel]')
    print('data:   ', data.get_shape().as_list())
    
    C1 = tf.nn.conv2d(data, C1_filter, [1, 1, 1, 1], padding='SAME')  ## C1: 6@28x28
    C1 = tf.nn.relu(C1 + C1_biases)
    print('C1:     ', C1.get_shape().as_list())
    
    S2 = tf.nn.max_pool(C1, [1,2,2,1], [1,2,2,1], padding='VALID')    ## S2: 6@14x14
    print('S2:     ', S2.get_shape().as_list())
    
    C3 = tf.nn.conv2d(S2, C3_filter, [1, 1, 1, 1], padding='VALID')   ## C3: 16@10x10
    C3 = tf.nn.relu(C3 + C3_biases)
    print('C3:     ', C3.get_shape().as_list())
    
    S4 = tf.nn.max_pool(C3, [1,2,2,1], [1,2,2,1], padding='VALID')    ## S4: 16@5x5
    print('S4:     ', S4.get_shape().as_list())
    
    shape = S4.get_shape().as_list()
    reshape = tf.reshape(S4, [shape[0], shape[1] * shape[2] * shape[3]])
    print('reshape:', reshape.get_shape().as_list())
    
    C5 = tf.nn.relu(tf.matmul(reshape, C5_weights) + C5_biases)
    print('C5:     ', C5.get_shape().as_list())
    
    F6 = tf.nn.relu(tf.matmul(C5, F6_weights) + F6_biases)
    print('F6:     ', F6.get_shape().as_list())
    
    OUTPUT = tf.matmul(F6, OUTPUT_weights) + OUTPUT_biases
    print('OUTPUT: ', OUTPUT.get_shape().as_list(), '\n\n')
    
    return OUTPUT
  
  print('TRAINING')    
  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
  # Predictions for the training, validation, and test data.

  train_prediction = tf.nn.softmax(logits)    
  print('VALIDATION')
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  print('TESTING')
  test_prediction = tf.nn.softmax(model(tf_test_dataset))
```

    TRAINING
             [batch, height, width, channel]
    data:    [16, 28, 28, 1]
    C1:      [16, 28, 28, 6]
    S2:      [16, 14, 14, 6]
    C3:      [16, 10, 10, 16]
    S4:      [16, 5, 5, 16]
    reshape: [16, 400]
    C5:      [16, 120]
    F6:      [16, 84]
    OUTPUT:  [16, 10] 
    
    
    VALIDATION
             [batch, height, width, channel]
    data:    [10000, 28, 28, 1]
    C1:      [10000, 28, 28, 6]
    S2:      [10000, 14, 14, 6]
    C3:      [10000, 10, 10, 16]
    S4:      [10000, 5, 5, 16]
    reshape: [10000, 400]
    C5:      [10000, 120]
    F6:      [10000, 84]
    OUTPUT:  [10000, 10] 
    
    
    TESTING
             [batch, height, width, channel]
    data:    [10000, 28, 28, 1]
    C1:      [10000, 28, 28, 6]
    S2:      [10000, 14, 14, 6]
    C3:      [10000, 10, 10, 16]
    S4:      [10000, 5, 5, 16]
    reshape: [10000, 400]
    C5:      [10000, 120]
    F6:      [10000, 84]
    OUTPUT:  [10000, 10] 
    
    


### Step 4: Run a TensorFlow session (`tf.session`) to train, validate and test the model


```python
num_steps = 1001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('\t',     'Minibatch\t', 'Minibatch\t',  'Validation')
  print('Step\t', 'Loss\t\t',      'Accuracy\t',  'Accuracy')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('%d\t %f\t %.1f%%\t\t %.1f%%\t' % (
            step,
            l,
            accuracy(predictions, batch_labels),
            accuracy(valid_prediction.eval(), valid_labels)
        ))
  print('*** TEST ACCURACY: %.1f%% ***' % accuracy(test_prediction.eval(), test_labels))
```

    	 Minibatch	 Minibatch	 Validation
    Step	 Loss		 Accuracy	 Accuracy
    0	 4.374419	 6.2%		 10.0%	
    50	 2.229358	 12.5%		 21.1%	
    100	 1.112568	 50.0%		 53.2%	
    150	 0.665704	 75.0%		 68.5%	
    200	 0.652371	 81.2%		 73.8%	
    250	 1.317888	 62.5%		 74.5%	
    300	 0.861638	 68.8%		 78.5%	
    350	 0.964509	 68.8%		 78.8%	
    400	 1.044873	 75.0%		 78.5%	
    450	 0.537551	 81.2%		 79.9%	
    500	 0.424862	 87.5%		 80.5%	
    550	 0.235435	 93.8%		 81.0%	
    600	 0.509160	 81.2%		 81.3%	
    650	 0.416043	 87.5%		 82.3%	
    700	 0.589775	 81.2%		 81.6%	
    750	 0.274672	 100.0%		 82.5%	
    800	 0.554814	 87.5%		 83.0%	
    850	 0.633812	 87.5%		 82.7%	
    900	 0.525710	 87.5%		 81.9%	
    950	 0.155540	 93.8%		 83.7%	
    1000	 0.205820	 100.0%		 83.5%	
    ***Test accuracy: 90.4%









## Appendix: Quick reference for some TensorFlow APIs

The convolutional model above uses convolutions with stride 2 to reduce the dimensionality. Replace the strides by a max pooling operation (`nn.max_pool()`) of stride 2 and kernel size 2.

    ##
    ## tf.nn.conv2d( 
    ##               input,            // [batch, in_height, in_width, in_channels] 
    ##               filter/kernel,    // [filter_height, filter_width, in_channels, out_channels]([16,16,1,16])
    ##               strides,          // [1, stride, stride, 1]（一步的大小）
    ##               padding           // 'VALID'(smaller output) of 'SAME'(auto zero padding!)
    ##              ) 
    ##              => 4d-tensor       // A deeper feature map as next input
    ##
    ## tf.nn.max_pool(
    ##               value,            // A 4-D Tensor with shape [batch, height, width, channels]
    ##               ksize,            // The size of the window for each dimension of the input tensor.
    ##               strides,          // The stride of the sliding window for each dimension of the input tensor.
    ##               padding,          // 'VALID' or 'SAME'
    ##              )
    ##              => 4d-Tensor

---
