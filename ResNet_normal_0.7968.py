import numpy as np
import tensorflow as tf


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def softmax_layer(inpt, shape):
    fc_w = weight_variable(shape)
    fc_b = tf.Variable(tf.zeros([shape[1]]))

    fc_h = tf.nn.softmax(tf.matmul(inpt, fc_w) + fc_b)

    return fc_h


def conv_layer(inpt, filter_shape, stride):
    out_channels = filter_shape[3]

    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    mean, var = tf.nn.moments(conv, axes=[0, 1, 2])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = weight_variable([out_channels], name="gamma")

    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)

    out = tf.nn.relu(batch_norm)

    return out


def residual_block(inpt, output_depth, down_sample, projection=False):
    input_depth = inpt.get_shape().as_list()[3]
    if down_sample:
        filter_ = [1, 2, 2, 1]
        inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')

    conv1 = conv_layer(inpt, [3, 3, input_depth, output_depth], 1)
    conv2 = conv_layer(conv1, [3, 3, output_depth, output_depth], 1)

    if input_depth != output_depth:
        if projection:
            # Option B: Projection shortcut
            input_layer = conv_layer(inpt, [1, 1, input_depth, output_depth], 2)
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(inpt, [[0, 0], [0, 0], [0, 0], [0, output_depth - input_depth]])
    else:
        input_layer = inpt

    res = conv2 + input_layer


    return res



n_dict = {20: 1, 32: 2, 44: 3, 56: 4}

def resnet(inpt, n):
    if n < 20 or (n - 20) % 12 != 0:
        print "ResNet depth invalid."
        return
    inpt = tf.reshape(inpt, shape=[-1, 28, 28, 1])
    num_conv = (n - 20) / 12 + 1
    layers = []

    with tf.variable_scope('conv1'):
        conv1 = conv_layer(inpt, [3, 3, 1, 16], 1)
        layers.append(conv1)

    for i in range(num_conv):
        with tf.variable_scope('conv2_%d' % (i + 1)):
            conv2_x = residual_block(layers[-1], 16, False)
            conv2 = residual_block(conv2_x, 16, False)
            layers.append(conv2_x)
            layers.append(conv2)

        assert conv2.get_shape().as_list()[1:] == [28, 28, 16]

    for i in range(num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv3_%d' % (i + 1)):
            conv3_x = residual_block(layers[-1], 32, down_sample)
            conv3 = residual_block(conv3_x, 32, False)
            layers.append(conv3_x)
            layers.append(conv3)

        assert conv3.get_shape().as_list()[1:] == [14, 14, 32]

    for i in range(num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv4_%d' % (i + 1)):
            conv4_x = residual_block(layers[-1], 64, down_sample)
            conv4 = residual_block(conv4_x, 64, False)
            layers.append(conv4_x)
            layers.append(conv4)

        assert conv4.get_shape().as_list()[1:] == [7, 7, 64]

    with tf.variable_scope('fc'):
        global_pool = tf.reduce_mean(layers[-1], [1, 2])
        assert global_pool.get_shape().as_list()[1:] == [64]

        out = softmax_layer(global_pool, [64, 10])
        layers.append(out)


    return layers[-1]




from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data1/", one_hot=True)


batch_size = 128
training_iters = 1500000
display_step = 20

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])
learning_rate = tf.placeholder("float", [])

# ResNet Models
net = resnet(X, 20)
# net = models.resnet(X, 32)
# net = models.resnet(X, 44)
# net = models.resnet(X, 56)

cross_entropy = -tf.reduce_sum(Y * tf.log(net))
opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
train_op = opt.minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver()
# checkpoint = tf.train.latest_checkpoint(".")
# if checkpoint:
#     print "Restoring from checkpoint", checkpoint
#     saver.restore(sess, checkpoint)
# else:
#     print "Couldn't find checkpoint to restore from. Starting over."

batch_xs, batch_ys = mnist.train.next_batch(batch_size)
batch_xs=np.reshape(batch_xs,[np.shape(batch_xs)[0],28,28,1])



for j in range(1):
    for i in range(0, training_iters, batch_size):
        feed_dict = {
            X: batch_xs,
            Y: batch_ys,
            learning_rate: 0.001}
        sess.run([train_op], feed_dict=feed_dict)

        # if i % 512 == 0:
        #     print "training on image #%d" % i
        #     saver.save(sess, 'progress', global_step=i)
        if i % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cross_entropy, accuracy], feed_dict=feed_dict)

            print("Iter " + str(i) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))


batch_testx, batch_testy = mnist.train.next_batch(batch_size)
batch_testx=np.reshape(batch_testx,[np.shape(batch_testx)[0],28,28,1])


acc = sess.run([accuracy], feed_dict={X: batch_testx, Y: batch_testy})

accuracy_summary = tf.summary.scalar("accuracy", accuracy)
print acc


sess.close()