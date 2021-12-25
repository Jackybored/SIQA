import numpy as np
import os
import PIL.Image as Image

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt
#the path of output data(tfrecords)
DATA_DIR="dataset_crop_images/"

#the path of input data(images,dmos)
DATABASE_DIR="D:\deep_learning\Live 3D"
MOS_PATH="D:/jupyter/3D图像质量评价/baseline/LIVE1_3DIQA.txt"
REF_PATH=os.path.join(DATABASE_DIR, "Phase1/")
#the info of the image
HEIGHT=360
WIDTH=640
DEPTH=1
#the info of the training process
TRAIN_DATA_NUM=16
TEST_DATA_NUM=4
NUM_PER_IMAGE =[18,19,16,17,17,18,20,18,20,17,20,17,18,19,19,17,20,18,18,19,0]
#the paras of dataset
SHUFFLE_SIZE=1000
NUM_EPOCH=50
BATCH_SIZE=1
BATCH_SIZE_TEST=1
#the info of the input tensor
PATCH_SIZE=32
NUM_PATCHES_PER_IMAGE=66




def load_image(path,gray_scale=False):
    """
    read the image from the path,and return a Image object
    """
    image=Image.open(path)
    if gray_scale:
        image=image.convert('L')
    else:
        image=image.convert("RGB")
    return image


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(images_l,images_r,images_l_grad,images_r_grad,images_dis,labels,filename,i):
    """
    convert the data to tfrecord
    """
    if not gfile.Exists(filename):
        print("writing",filename)
        writer=tf.python_io.TFRecordWriter(filename)
        for index in range(NUM_PER_IMAGE[i]):
            image_l=images_l[index].tostring()
            image_r=images_r[index].tostring()
            image_l_grad=images_l_grad[index].tostring()
            image_r_grad=images_r_grad[index].tostring()
            image_dis=images_dis[index].tostring()
            feature={
                "label":_float_feature(labels[index]),
                "image_l":_bytes_feature(image_l),
                "image_r":_bytes_feature(image_r),
                "image_dis":_bytes_feature(image_dis),
                "image_l_grad":_bytes_feature(image_l_grad),
                "image_r_grad":_bytes_feature(image_r_grad)
            }
            features=tf.train.Features(feature=feature)
            example=tf.train.Example(features=features)
            writer.write(example.SerializeToString())
        writer.close()



def load_data():
    """
    load data, and convert the data to tfrecords
    """
    text_file = open(MOS_PATH, "r")
    lines = text_file.readlines()
    text_file.close()
    print(len(lines))
    # read the image of left and right and the dmos
    L_img_set = []
    R_img_set = []
    L_grad_img_set = []
    R_grad_img_set = []
    disparity_set = []
    dmos_set = []
    for line in lines:
        read_line = line.rstrip().split(" ")
        left_name = read_line[1]
        right_name = read_line[2]
        disparity_name = read_line[3]
        left_grad_name = read_line[4]
        right_grad_name = read_line[5]
        dmos = read_line[6]
        path_left = REF_PATH + left_name.lower()
        path_right = REF_PATH + right_name.lower()
        path_disparity = REF_PATH + "disparity\\" + disparity_name.lower()
        path_left_grad = REF_PATH + "gradient\\" + left_grad_name
        path_right_grad = REF_PATH + "gradient\\" + right_grad_name
        L_img_set.append(np.reshape(np.asarray(load_image(path_left, gray_scale=False), dtype=np.uint8),
                                    HEIGHT * WIDTH * DEPTH))  # ndarray(0,255)
        R_img_set.append(
            np.reshape(np.asarray(load_image(path_right, gray_scale=False), dtype=np.uint8), HEIGHT * WIDTH * DEPTH))
        disparity_set.append(
            np.reshape(np.asarray(load_image(path_disparity, gray_scale=True), dtype=np.uint8), HEIGHT * WIDTH))
        L_grad_img_set.append(np.reshape(np.asarray(load_image(path_left_grad, gray_scale=False), dtype=np.uint8),
                                         HEIGHT * WIDTH * DEPTH))
        R_grad_img_set.append(np.reshape(np.asarray(load_image(path_right_grad, gray_scale=False), dtype=np.uint8),
                                         HEIGHT * WIDTH * DEPTH))
        dmos_set.append(float(dmos))
    FRONT = 0
    BACK = NUM_PER_IMAGE[0]
    # convert the data to tfrecord
    for i in range(TRAIN_DATA_NUM + TEST_DATA_NUM):
        images_l = [L_img_set[j] for j in range(FRONT, BACK)]
        images_r = [R_img_set[j] for j in range(FRONT, BACK)]
        images_dis = [disparity_set[j] for j in range(FRONT, BACK)]
        images_l_grad = [L_grad_img_set[j] for j in range(FRONT, BACK)]
        images_r_grad = [R_grad_img_set[j] for j in range(FRONT, BACK)]
        labels = [dmos_set[j] for j in range(FRONT, BACK)]

        FRONT = BACK
        BACK = BACK + NUM_PER_IMAGE[i + 1]

        if not gfile.Exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        filename = os.path.join(DATA_DIR, "image_" + str(i) + ".tfrecords")
        convert_to_tfrecord(images_l, images_r, images_l_grad, images_r_grad, images_dis, labels, filename, i)


def parser(record):
    """
    parse the tfrecords
    """
    features = tf.parse_single_example(
        record,
        features={
            "label": tf.FixedLenFeature([], tf.float32),
            "image_l": tf.FixedLenFeature([], tf.string),
            "image_r": tf.FixedLenFeature([], tf.string)
        })
    image_l = tf.decode_raw(features["image_l"], tf.uint8)
    image_r = tf.decode_raw(features["image_r"], tf.uint8)
    image_l=tf.cast(image_l,tf.float32)*(1./255)-0.5
    image_r=tf.cast(image_r,tf.float32)*(1./255)-0.5
    image_l = tf.reshape(image_l, [HEIGHT, WIDTH, DEPTH])
    image_r = tf.reshape(image_r, [HEIGHT, WIDTH, DEPTH])
    label = features["label"]
    return image_l, image_r,label


def parser_test(record):
    """
    parse the tfrecords
    """
    features = tf.parse_single_example(
        record,
        features={
            "label": tf.FixedLenFeature([], tf.float32),
            "image_l": tf.FixedLenFeature([], tf.string),
            "image_r": tf.FixedLenFeature([], tf.string)
        })
    image_l = tf.decode_raw(features["image_l"], tf.uint8)
    image_r = tf.decode_raw(features["image_r"], tf.uint8)
    image_l=tf.cast(image_l,tf.float32)*(1./255)-0.5
    image_r=tf.cast(image_r,tf.float32)*(1./255)-0.5
    image_l = tf.reshape(image_l, [NUM_PATCHES_PER_IMAGE, PATCH_SIZE, PATCH_SIZE, DEPTH])
    image_r = tf.reshape(image_r, [NUM_PATCHES_PER_IMAGE, PATCH_SIZE, PATCH_SIZE, DEPTH])
    label = features["label"]
    return image_l, image_r,label

#concat the disparity to the image
def add_disparity(images_l,images_r,images_dis,images_l_grad,images_r_grad):
    images_dis_l=tf.concat([images_l,images_dis],axis=3)
    images_dis_r=tf.concat([images_r,images_dis],axis=3)
    images_dis_l_grad=tf.concat([images_l_grad,images_dis],axis=3)
    images_dis_r_grad=tf.concat([images_r_grad,images_dis],axis=3)
    return images_dis_l,images_dis_r,images_dis_l_grad,images_dis_r_grad

#construct the patches of each image
def random_sample(images_l, images_r, images_l_grad, images_r_grad, patch_size, num_patches):
    """
    random sample the patch pairs from image pairs
    """
    # tf.set_random_seed(seed)
    batch_size=BATCH_SIZE
    with tf.variable_scope('patches_extract'):
        patch_l = []
        patch_r = []
        patch_l_grad = []
        patch_r_grad = []

        images_lr = tf.concat([images_l, images_r], axis=3)
        images_lr_grad = tf.concat([images_l_grad, images_r_grad], axis=3)
        for i in range(batch_size):
            for j in range(num_patches):
                patch_lr = tf.random_crop(images_lr[i, :, :, :],
                                          [patch_size, patch_size, images_lr.get_shape()[3].value])
                patch_l.append(patch_lr[:, :, 0:3])
                patch_r.append(patch_lr[:, :, 3:6])

                patch_lr_grad = tf.random_crop(images_lr_grad[i, :, :, :],
                                               [patch_size, patch_size, images_lr_grad.get_shape()[3].value])
                patch_l_grad.append(patch_lr_grad[:, :, 0:3])
                patch_r_grad.append(patch_lr_grad[:, :, 3:6])

        patch_l = tf.convert_to_tensor(value=patch_l, dtype=tf.float32, name="sample_patches_l")
        patch_r = tf.convert_to_tensor(value=patch_r, dtype=tf.float32, name="sample_patches_r")
        patch_l_grad = tf.convert_to_tensor(value=patch_l_grad, dtype=tf.float32, name="sample_patches_l_grad")
        patch_r_grad = tf.convert_to_tensor(value=patch_r_grad, dtype=tf.float32, name="sample_patches_r_grad")
        return patch_l, patch_r, patch_l_grad, patch_r_grad

def distored_input(filenames):

    with tf.variable_scope("input_data"):
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(parser_test)
        dataset = dataset.shuffle(buffer_size=SHUFFLE_SIZE)
        dataset = dataset.repeat(NUM_EPOCH)
        dataset = dataset.batch(BATCH_SIZE)
        iterator = dataset.make_one_shot_iterator()
        images_l, images_r, labels = iterator.get_next()
        # patches_l, patches_r, patches_l_grad, patches_r_grad = random_sample(images_l, images_r,
        #                                                                      images_l_grad, images_r_grad,
        #                                                                      patch_size=PATCH_SIZE,
        #                                                                      num_patches=NUM_PATCHES_PER_IMAGE)
        patches_l = tf.reshape(images_l, [BATCH_SIZE * NUM_PATCHES_PER_IMAGE, PATCH_SIZE, PATCH_SIZE, DEPTH])
        patches_r = tf.reshape(images_r, [BATCH_SIZE * NUM_PATCHES_PER_IMAGE, PATCH_SIZE, PATCH_SIZE, DEPTH])
        return patches_l, patches_r,labels

def distored_input_test(filenames):
    with tf.variable_scope("input_data_test"):
        dataset=tf.data.TFRecordDataset(filenames)
        dataset=dataset.map(parser_test)
        dataset=dataset.batch(BATCH_SIZE_TEST)
        iterator=dataset.make_one_shot_iterator()
        images_l, images_r, labels = iterator.get_next()
        images_l = tf.reshape(images_l, [BATCH_SIZE_TEST * NUM_PATCHES_PER_IMAGE, PATCH_SIZE, PATCH_SIZE, DEPTH])
        images_r = tf.reshape(images_r, [BATCH_SIZE_TEST * NUM_PATCHES_PER_IMAGE, PATCH_SIZE, PATCH_SIZE, DEPTH])

        return  images_l,images_r,labels