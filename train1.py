import numpy as np
import os

import tensorflow as tf
import PI3D2013
import set_tfrecord
from scipy import stats
import scipy.io as sio
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile
import warnings
import time as TIME
HEIGHT=360
WIDTH=640
DEPTH=3
#the info of the training process
TRAIN_DATA_NUM=16
TEST_DATA_NUM=4
NUM_PER_IMAGE = [18,19,16,17,17,18,20,18,20,17,20,17,18,19,19,17,20,18,18,19,0]

#the path which save the model
BEST_MODEL="logs/best/mid_30/"
# BEST_MODEL="logs/pretrain/"
TEMP_MODEL="logs/temp/"
PRETRAIN_MODEL="logs/pretrain/"

train_dir = './logs/events/train/'
test_dir="./logs/events/"
#the dataset path
# TFRecord_path_train = "grad_dataset_new_dis/"
TFRecord_path_train = "dataset/selected_cyc_mscn_med_30/"
TFRecord_path_test="dataset/selected_cyc_mscn_med_30/"
#禁止显示运行设备等
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL']="3"
def train(seed,time):
    best_srocc=-5
    best_plcc=0
    graph=tf.Graph()
    with graph.as_default():
        #set the global_step and keep_prob for training
        global_step = tf.Variable(0, trainable=False)
        keep_prob = tf.placeholder(tf.float32, name='ratio')
        #Input images and labels
        filenames = [os.path.join(TFRecord_path_train, "image_" + str(i) + ".tfrecords") for i in ORDER[0:TRAIN_DATA_NUM]]
        patches_l, patches_r,labels = set_tfrecord.distored_input(filenames)
        # the predicted score of each image
        scores = PI3D2013.inference(patches_l, patches_r, keep_prob,type="train")
        #the loss of the prediction
        total_loss = PI3D2013.loss_func(scores, labels)
        # Add to the Graph the
        #
        # Ops that calculate and apply gradients.
        Number = 0
        for i in range(TRAIN_DATA_NUM):
            Number += NUM_PER_IMAGE[ORDER[i]]
        train_op,lr = PI3D2013.train_func(data_num=Number, total_loss=total_loss,
                                        global_step=global_step)
        # Create a saver for writing training checkpoints.

        # restore_var=[v for v in tf.global_variables()]
        # variable_to_restore=[v for v in restore_var if v.name.startswith("all_layers")]
        # saver_restore = tf.train.Saver(var_list=variable_to_restore)
        # checkpoint_file = os.path.join(PRETRAIN_MODEL + "model" + str(time) + str(seed) + "/", 'pre_train_model.ckpt')

        saver=tf.train.Saver()
        #the all steps for training
        MAX_STEPS = int(set_tfrecord.NUM_EPOCH * Number / set_tfrecord.BATCH_SIZE)
        print(MAX_STEPS)
        iters_per_epoch = Number//set_tfrecord.BATCH_SIZE
        print(iters_per_epoch)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    with tf.Session(graph=graph,config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        #Initialize the variables
        sess.run(tf.global_variables_initializer())
        # saver_restore.restore(sess,checkpoint_file)
        # print(sess.run(global_step))
        # print(sess.run(lr))
        for step in range(MAX_STEPS):
            try:
                _,learning_rate, loss_value,steps = sess.run([train_op,lr,total_loss,global_step], feed_dict={keep_prob: 0.5})
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            except tf.errors.OutOfRangeError:
                break

            if step % (2*iters_per_epoch) == 0 or (step+1) == MAX_STEPS:
                # Save a checkpoint and evaluate the model periodically.
                checkpoint_file = os.path.join(TEMP_MODEL, 'temp_model.ckpt')
                saver.save(sess, checkpoint_file)
                test_loss,srocc,plcc = evaluate('test',seed,time)

                if srocc > best_srocc:
                    best_srocc=srocc
                    best_plcc=plcc
                    best_epoch = step / iters_per_epoch
                    print('best epoch %d with min loss %.3f' % (best_epoch, test_loss))

                    checkpoint_file = os.path.join(PRETRAIN_MODEL+ "model"+ str(time)+str(seed)+"/", 'pre_train_model.ckpt')
                    saver.save(sess, checkpoint_file)

            if step % (10 * iters_per_epoch) == 0 or (step + 1) == MAX_STEPS:
                train_loss,train_srocc,train_plcc = evaluate("train",seed,time)
                print('Epoch %d (Step %d): train_loss = %.3f' % (step / iters_per_epoch, step, train_loss))

    return best_srocc, best_plcc

def evaluate(type,seed,time,isSave=False):
    graph = tf.Graph()
    with graph.as_default() as g:
        keep_prob = tf.placeholder(tf.float32, name="ratio")
        num_data = 0
        if type=="train":
            filenames = [os.path.join(TFRecord_path_test, "image_" + str(i) + ".tfrecords") for i in ORDER[:TRAIN_DATA_NUM]]
            for i in range(TRAIN_DATA_NUM):
                num_data += NUM_PER_IMAGE[ORDER[i]]
        if type=="test":
            filenames = [os.path.join(TFRecord_path_test, "image_" + str(i) + ".tfrecords") for i in ORDER[TRAIN_DATA_NUM:TRAIN_DATA_NUM+TEST_DATA_NUM]]
            for i in range(TEST_DATA_NUM):
                num_data += NUM_PER_IMAGE[ORDER[TRAIN_DATA_NUM + i]]
        if type=="best_test":
            filenames = [os.path.join(TFRecord_path_test, "image_" + str(i) + ".tfrecords") for i in ORDER[TRAIN_DATA_NUM:TRAIN_DATA_NUM+TEST_DATA_NUM]]
            for i in range(TEST_DATA_NUM):
                num_data += NUM_PER_IMAGE[ORDER[TRAIN_DATA_NUM + i]]
        #get the images and labels which to test
        patches_l, patches_r, labels = set_tfrecord.distored_input_test(filenames)
        #the scores of images
        scores = PI3D2013.inference(patches_l, patches_r,keep_prob,type="test")
        #the loss of images
        total_loss=PI3D2013.loss_func(scores, labels)
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            PI3D2013.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)


    with tf.Session(graph=graph) as sess:
        if type == 'test':
            checkpoint_file = os.path.join(TEMP_MODEL, 'temp_model.ckpt')
        if type=="best_test":
            checkpoint_file = os.path.join(BEST_MODEL+ "model"+ str(time)+str(seed)+"/", 'pre_train_model.ckpt')
        if type=="train":
            checkpoint_file = os.path.join(PRETRAIN_MODEL + "model" + str(time) + str(seed) + "/", 'pre_train_model.ckpt')
        saver.restore(sess, checkpoint_file)

        score_set = []
        label_set = []
        loss_set = []
        step = 0
        num_iter = num_data
        #compute the scores of each image
        while step < num_iter//set_tfrecord.BATCH_SIZE_TEST:
            loss_hat, scores_hat, labels_hat = sess.run([total_loss, scores, labels], feed_dict={keep_prob: 1.0})
            score_set.append(scores_hat)
            label_set.append(labels_hat)
            loss_set.append(loss_hat)
            step += 1

        score_set = np.reshape(np.asarray(score_set), (-1,))
        label_set = np.reshape(np.asarray(label_set), (-1,))
        loss_set = np.reshape(np.asarray(loss_set), (-1,))
        if isSave:
            save_pred="obj.mat"
            pred_array=np.reshape(score_set,[len(score_set),1])
            sio.savemat(save_pred,{"pred_dmos":pred_array})
            save_real="sub.mat"
            real_array=np.reshape(label_set,[len(label_set),1])
            sio.savemat(save_real,{"real_dmos":real_array})

        # Compute evaluation metric.
        mae = loss_set.mean()
        srocc = stats.spearmanr(score_set, label_set)[0]
        krocc = stats.stats.kendalltau(score_set, label_set)[0]
        plcc = stats.pearsonr(score_set, label_set)[0]
        rmse = np.sqrt(((score_set - label_set) ** 2).mean())
        mse = ((score_set - label_set) ** 2).mean()
        print("%s: MAE: %.3f\t SROCC: %.3f\t KROCC: %.3f\t PLCC: %.3f\t RMSE: %.3f\t MSE: %.3f"
              % (type, mae, srocc, krocc, plcc, rmse, mse))

    return mae,srocc,plcc


med_srocc=[]
med_plcc=[]
# times=[0,5,6,12,16,17,23,27,31]

# [0, 11, 4, 15, 7, 12, 8, 10, 17, 1, 19, 2, 3, 16, 13, 9, 18, 5, 14, 6]
samples=[
[2, 5, 17, 19, 12, 1, 11, 10, 13, 18, 7, 4, 8, 9, 0, 16, 6, 15, 14, 3],
[17, 7, 5, 6, 8, 2, 15, 14, 4, 11, 12, 1, 18, 0, 16, 13, 19, 3, 9, 10],
[19, 13, 9, 7, 14, 16, 10, 8, 5, 4, 1, 18, 0, 12, 15, 3, 2, 17, 6, 11],
[3, 17, 16, 15, 12, 7, 6, 2, 8, 10, 0, 19, 18, 13, 4, 11, 1, 5, 14, 9],
[14, 3, 2, 13, 5, 8, 12, 1, 10, 11, 7, 4, 18, 0, 9, 16, 6, 19, 17, 15],
[0, 10, 2, 1, 4, 5, 14, 15, 3, 16, 7, 11, 12, 13, 17, 18, 9, 8, 6, 19],
[7, 15, 0, 11, 17, 3, 6, 4, 14, 12, 10, 2, 1, 13, 9, 18, 5, 16, 8, 19],
[15, 0, 5, 13, 8, 4, 3, 7, 1, 9, 11, 17, 19, 14, 10, 12, 6, 2, 16, 18],
[4, 18, 5, 6, 16, 17, 8, 15, 3, 14, 13, 10, 7, 11, 2, 0, 9, 1, 19, 12],
[0, 11, 4, 15, 7, 12, 8, 10, 17, 1, 19, 2, 3, 16, 13, 9, 18, 5, 14, 6]]

for time in [9]:
    temp_srocc=[]
    temp_plcc=[]
    for seed in [3]:
        # ORDER = np.random.permutation(TRAIN_DATA_NUM + TEST_DATA_NUM)
        start=TIME.clock()
        ORDER=samples[time]
        # print(ORDER)
        mae,srocc,plcc=evaluate("best_test",seed,time,isSave=False)
        # srocc,plcc=train(seed,time)
        end=TIME.clock()
        print(end-start)
        temp_srocc.append(srocc)
        temp_plcc.append(plcc)
        print(temp_srocc)
        print(temp_plcc)
        print(np.median(np.asarray(temp_srocc)))
        print(np.median(np.asarray(temp_plcc)))
    med_srocc.append(temp_srocc)
    med_plcc.append(temp_plcc)
print(med_srocc)
print(med_plcc)
print(np.median(np.asarray(med_srocc)))
print(np.median(np.asarray(med_plcc)))