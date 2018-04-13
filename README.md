# DNN-for-fashion-mnist
Fashion test accuracy 0.9896

# Request
        tensorflow 1.0+
        python 2.76
        CUDA 7.0
        fashion mnist data

# Network Structure

![](https://github.com/SrCMpink/HelloWorld/blob/master/DNN-net1.png)  
The network structure is based off tf-tutorials-mnist-examples with the addition of Conv and Dropout. I modify the convolution kernel of the example network from 5*5 to 3*3 and adde a convolution layer but do not add a maxpooling layer.Dropout is used between fully connected layers. The loss function is cross entropy.The network training uses adam's optimization algorithm.The number of network training iterations is 1.5 million.

        The detailed process of training is shown in the figure below

# Training 

![](https://github.com/SrCMpink/HelloWorld/blob/master/DNN-net-train.png) 

    training_iters = 1500000
    batch_size = 100
    learning_rate = 0.001
    dropout1 = 0.75
    dropout2 = 0.75
    padding='SAME'
    cost = tf.nn.softmax_cross_entropy_with_logits
    optimizer = tf.train.AdamOptimizer
    

