"""
@File         : tid2013.py
@Time         : 2017/6/22 
@Author       : Chen Huang
@Update       : 
@Discription  : Builds the network.
"""




import numpy
import tensorflow as tf

# import PI3D2013_input
import set_tfrecord
# training parameters
INI_LEARNING_RATE = 0.001 #0.0001

MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 5 # 350.0  # Epochs after which learning rate decays.  #5000
LEARNING_RATE_DECAY_FACTOR = 0.8 #Learning rate decay factor. #0.1

EPSILON = 0.000001

DATA_TYPE = tf.float32


def _activation_summary(x):
    tf.summary.histogram("actiation",x)
    tf.summary.scalar("sparsity",tf.nn.zero_fraction(x))


def conv_layer(input, kernel_shape, output_channels, activation=None):
    """ conv + relu
    
    Args:
        :param input: tensor. 
        :param kernel_shape: tuple - (height, width).
        :param output_channels: int.
        :param activation: func.
        
    Returns:
        :return output: tensor.
    """
    input_channel = input.get_shape()[3].value
    weights_initializer = tf.truncated_normal_initializer(
        stddev=1.0 / numpy.sqrt(float(kernel_shape[0]*kernel_shape[1]*input_channel)))
    weights = tf.get_variable(name='weights',
                              shape=kernel_shape + (input_channel, output_channels),
                              initializer=weights_initializer, dtype=tf.float32)
    biases = tf.get_variable(name='biases', shape=[output_channels],
                             initializer=tf.constant_initializer(0.0), dtype=tf.float32)

    # regularizer=tf.contrib.layers.l2_regularizer(0.0001)
    # tf.add_to_collection("losses",regularizer(weights))

    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')+biases
    if activation==None:
        output=conv
    else:
        output = activation(conv)

    return output

def ACB_layer(input, output_channels):
    with tf.variable_scope("ACB_layer"):
        with tf.variable_scope("layer1"):
            layer1=conv_layer(input,kernel_shape=(3,3),output_channels=output_channels,activation=tf.nn.relu)
        with tf.variable_scope("layer2"):
            layer2=conv_layer(input,kernel_shape=(1,3),output_channels=output_channels,activation=tf.nn.relu)
        with tf.variable_scope("layer3"):
            layer3=conv_layer(input,kernel_shape=(3,1),output_channels=output_channels,activation=tf.nn.relu)

    return layer1+layer2+layer3

def extract_features_layer(patches):
    with tf.variable_scope("extract_pre_merge_features"):
        with tf.variable_scope("block1"):
            with tf.variable_scope("layer1"):
                layer1_1_conv = conv_layer(patches, kernel_shape=(3, 3), output_channels=32, activation=tf.nn.relu)
                # layer1_1_conv=ACB_layer(patches, output_channels=32)
            with tf.variable_scope("layer2"):
                layer1_2_conv=conv_layer(layer1_1_conv,kernel_shape=(3,3),output_channels=32,activation=tf.nn.relu)
                # layer1_2_conv = ACB_layer(layer1_1_conv, output_channels=32)
                layer1_2_pool = tf.nn.max_pool(value=layer1_2_conv, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME",
                                             name="layer2_pool")
        with tf.variable_scope("block2"):
            with tf.variable_scope("layer1"):
                layer2_1_conv=conv_layer(layer1_2_pool,kernel_shape=(3,3),output_channels=64,activation=tf.nn.relu)
                # layer2_1_conv = ACB_layer(layer1_2_pool, output_channels=64)
            with tf.variable_scope("layer2"):
                layer2_2_conv=conv_layer(layer2_1_conv,kernel_shape=(3,3),output_channels=64,activation=tf.nn.relu)
                # layer2_2_conv = ACB_layer(layer2_1_conv, output_channels=64)

    return layer2_2_conv

def merge_layer(patches_x,patches_y):
    with tf.variable_scope("extract_init_features") as scope:
        feature_x=extract_features_layer(patches_x)
        scope.reuse_variables()
        feature_y=extract_features_layer(patches_y)
        merge_map=tf.concat([feature_x[:,:,:,:32], feature_y[:,:,:,:32],feature_x[:,:,:,32:64], feature_y[:,:,:,32:64]], axis=3)

    return merge_map,feature_x,feature_y

def diff_pool_layer(merge_map):
    with tf.variable_scope("pool1"):
        pool1=tf.nn.max_pool(merge_map,ksize=(1,2,2,1),strides=(1,2,2,1),padding="SAME")
        pool1_conv1=conv_layer(pool1,kernel_shape=(3,3),output_channels=128,activation=tf.nn.relu)
        # pool1_conv1 = ACB_layer(pool1, output_channels=128)
        pool1_pool=tf.nn.max_pool(pool1_conv1,ksize=(1,2,2,1),strides=(1,2,2,1),padding="SAME")

    with tf.variable_scope("pool2"):
        pool2=tf.nn.max_pool(merge_map,ksize=(1,4,4,1),strides=(1,4,4,1),padding="SAME")
        pool2_conv=conv_layer(pool2,kernel_shape=(3,3),output_channels=128,activation=tf.nn.relu)
        # pool2_conv = ACB_layer(pool2, output_channels=128)

    with tf.variable_scope("pool3"):
        pool3=tf.nn.max_pool(merge_map,ksize=(1,8,8,1),strides=(1,8,8,1),padding="SAME")
        pool3_conv=conv_layer(pool3,kernel_shape=(1,1),output_channels=256,activation=tf.nn.relu)
        # pool2_conv = ACB_layer(pool2, output_channels=128)
    return pool1_pool,pool2_conv,pool3_conv

def extract_single_features(patches):
    with tf.variable_scope("extract_single_features"):
        with tf.variable_scope("block1"):
            pool_layer=tf.nn.max_pool(patches,ksize=(1,2,2,1),strides=(1,2,2,1),padding="SAME",name="pool_layer")
            with tf.variable_scope("layer1"):
                # layer1_1_conv=conv_layer(pool_layer,kernel_shape=(3,3),output_channels=128,activation=tf.nn.relu)
                layer1_1_conv = ACB_layer(pool_layer, output_channels=128)
                layer1_1_pool = tf.nn.max_pool(value=layer1_1_conv, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME",
                                             name="layer1_pool")
        with tf.variable_scope("block2"):
            with tf.variable_scope("layer1"):
                # layer2_1_conv=conv_layer(layer1_1_pool,kernel_shape=(3,3),output_channels=256,activation=tf.nn.relu)
                layer2_1_conv = ACB_layer(layer1_1_pool, output_channels=256)
                layer2_1_pool = tf.nn.max_pool(value=layer2_1_conv, ksize=(1, 4, 4, 1), strides=(1, 4, 4, 1),
                                               padding="SAME",
                                               name="layer1_pool")
        # with tf.variable_scope("block3"):
        #     with tf.variable_scope("layer1"):
        #         # layer3_1_conv=conv_layer(layer2_1_pool,kernel_shape=(3,3),output_channels=512,activation=tf.nn.relu)
        #         layer3_1_conv = ACB_layer(layer2_1_pool, output_channels=512)
        #         layer3_1_pool = tf.nn.max_pool(value=layer3_1_conv, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
        #                                        padding="SAME",
        #                                        name="layer1_pool")
        return layer2_1_pool

# def extract_merge1_features(concat_map1):
#     with tf.variable_scope("merge1_features"):
#         layer1_conv=conv_layer(concat_map1,kernel_shape=(3,3),output_channels=256,activation=tf.nn.relu)
#         layer1_pool=tf.nn.max_pool(value=layer1_conv, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME",
#                                          name="layer1_pool")
#     return layer1_pool
# #
# def extract_merge2_features(concat_map2):
#     with tf.variable_scope("merge2_features"):
#         layer1_conv=conv_layer(concat_map2,kernel_shape=(3,3),output_channels=512,activation=tf.nn.relu)
#         layer1_pool=tf.nn.max_pool(value=layer1_conv, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME",
#                                          name="layer1_pool")
#     return layer1_pool

def extract_merge_features(concat_map):
    with tf.variable_scope("merge_features"):
        with tf.variable_scope("layer1"):
            layer1_conv=conv_layer(concat_map,kernel_shape=(3,3),output_channels=256,activation=tf.nn.relu)
            # layer1_conv = ACB_layer(concat_map, output_channels=256)
            layer1_pool=tf.nn.max_pool(value=layer1_conv, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME",
                                         name="layer1_pool")
        # with tf.variable_scope("layer2"):
        #     layer2_conv=conv_layer(layer1_pool,kernel_shape=(3,3),output_channels=512,activation=tf.nn.relu)
        #     # layer2_conv = ACB_layer(layer1_pool, output_channels=512)
        #     layer2_pool=tf.nn.max_pool(value=layer2_conv, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME",
        #                                  name="layer2_pool")
    return layer1_pool


def fully_connection(input, hiden_units1, hiden_units2, keep_prob):
    """ fully connection architecture.
    
    Args:
        :param input: tensor - (batch_size, ?)
        :param hiden_units: int.
        :param keep_prob: tensor. scalar.
    
    Returns:
        :return: tensor - (batch_size, 1)
    """
    input_units = input.get_shape()[1].value
    weights_initializer = tf.truncated_normal_initializer(stddev=1.0 / numpy.sqrt(float(input_units)))
    with tf.variable_scope('block1'):
        weights1 = tf.get_variable(name='weights1',
                                  shape=[input_units, hiden_units1],
                                  initializer=weights_initializer, dtype=DATA_TYPE)
        biases1 = tf.get_variable(name='biases1', shape=[hiden_units1],
                                 initializer=tf.constant_initializer(0.0), dtype=DATA_TYPE)
        full_conn1=tf.matmul(input,weights1)+biases1
        hidden1 = tf.nn.relu(full_conn1)
        hidden1_dropout = tf.nn.dropout(hidden1, keep_prob)

    with tf.variable_scope('block2'):
        weights2 = tf.get_variable(name='weights2',
                                  shape=[hiden_units1, hiden_units2],
                                  initializer=weights_initializer, dtype=DATA_TYPE)
        biases2 = tf.get_variable(name='biases2', shape=[hiden_units2],
                                 initializer=tf.constant_initializer(0.0), dtype=DATA_TYPE)
        full_conn2=tf.matmul(hidden1_dropout,weights2)+biases2
        hidden2 = tf.nn.relu(full_conn2)
        hidden2_dropout = tf.nn.dropout(hidden2, keep_prob)

    with tf.variable_scope('block3'):
        weights = tf.get_variable(name='weights',
                                  shape=[hiden_units2, 1],
                                  initializer=weights_initializer, dtype=DATA_TYPE)
        biases = tf.get_variable(name='biases', shape=[1],
                                 initializer=tf.constant_initializer(0.0), dtype=DATA_TYPE)
        output = tf.matmul(hidden2_dropout, weights) + biases

    return output

def inference(patches_x, patches_y, keep_prob, type):


    """Build the model up to where it may be used for inference.

    Args:
        :param patches_x, patches_y: tensor - (batch_size*num_patches_per_image, patch_size, patch_size, depth).
                Images placeholder, from inputs().
        :param keep_prob: tensor. scalar. used for fully_connection().

    Returns:
        :return output: tensor - (batch_size, )
    """
    with tf.variable_scope("all_layers"):
        if type=="train":
            batch_size=set_tfrecord.BATCH_SIZE
        else:
            batch_size=set_tfrecord.BATCH_SIZE_TEST
        with tf.variable_scope("extract_multi_features"):
            merge_map,feature_x,feature_y = merge_layer(patches_x,patches_y)
            merge1_pool,merge2_pool,merge3_pool=diff_pool_layer(merge_map)
            concat_map1=tf.concat([merge1_pool,merge2_pool],axis=3)
            # concat_map1_features=extract_merge1_features(concat_map1)
            # concat_map2=tf.concat([concat_map1_features,merge3_pool],axis=3)
            # merge_features=extract_merge2_features(concat_map2)
            concat_map1_features=extract_merge_features(concat_map1)
            flat1=tf.reshape(concat_map1_features,[set_tfrecord.NUM_PATCHES_PER_IMAGE * batch_size, -1])
            with tf.variable_scope("single_map") as scope:
                extracted_feature_x=extract_single_features(feature_x)
                scope.reuse_variables()
                extracted_feature_y=extract_single_features(feature_y)
            flat2=tf.reshape(extracted_feature_x,[set_tfrecord.NUM_PATCHES_PER_IMAGE * batch_size, -1])
            flat3=tf.reshape(extracted_feature_y,[set_tfrecord.NUM_PATCHES_PER_IMAGE * batch_size, -1])
            features_quality=tf.concat([flat1,flat2,flat3],axis=1)


        hiden_units1 = 512
        hiden_units2= 256
        with tf.variable_scope('regression'):
            patches_qualities = fully_connection(features_quality, hiden_units1, hiden_units2,keep_prob)

        with tf.variable_scope('weighting'):
            patches_weights = fully_connection(features_quality, hiden_units1,hiden_units2, keep_prob)
            patches_weights = tf.nn.relu(patches_weights)
            patches_weights = tf.reshape(patches_weights, (batch_size, set_tfrecord.NUM_PATCHES_PER_IMAGE)) + EPSILON
            patches_weights_sum = tf.expand_dims(tf.reduce_sum(patches_weights, axis=1), -1)
            patches_normalized_weights = patches_weights / patches_weights_sum

        with tf.variable_scope('estimate_quality'):
            patches_qualities = tf.reshape(patches_qualities, (batch_size, set_tfrecord.NUM_PATCHES_PER_IMAGE))
            patches_qualities = patches_qualities*patches_normalized_weights
            output = tf.reduce_sum(patches_qualities, axis=1)

    return output


def loss_func(scores, labels):
    """Mean absolute error.
    Args:
        :param scores: scores from inference().
        :param labels: Labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]

    Returns:
        :return Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    with tf.variable_scope('Loss'):
        # loss = tf.reduce_mean(tf.abs(scores - labels))
        loss=tf.reduce_mean((scores-labels)**2)
        tf.add_to_collection('losses', loss)
        # tf.summary.scalar(loss.op.name + '_raw', loss)
        return tf.add_n(tf.get_collection('losses'), name='total_loss')
        # return loss

def _add_loss_summaries(total_loss):
    """Add summaries for losses.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
        :param total_loss: Total loss from loss().

    Returns:
        :return loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + '_raw', l)
        tf.summary.scalar(l.op.name + '_average', loss_averages.average(l))

    return loss_averages_op


def train_func(data_num, total_loss, global_step):
    # Variables that affect learning rate.
    num_batches_per_epoch = data_num // set_tfrecord.BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INI_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    # tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    # loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.

    opt = tf.train.AdamOptimizer(learning_rate=lr, name='optimizer')
    grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # # Add histograms for trainable variables.
    # for var in tf.trainable_variables():
    #     tf.summary.histogram(var.op.name, var)
    #
    # # Add histograms for gradients.
    # for grad, var in grads:
    #     if grad is not None:
    #         tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op,lr
