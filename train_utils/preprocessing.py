import tensorflow as tf
from tensorflow import keras
import numpy as np

def normalize_image(image,label):
    return tf.cast(image,tf.float32) / 255., label

def resize_img(img,label):
    img = tf.image.resize_with_pad(img,target_height=224,target_width=224)
    return img,label

def imgnt_mean_substract1(image,label):
    #print(type(image))
    #print(image.shape)
    mean = [103.939, 116.779, 123.68]
    image[..., 0] -= mean[0]
    image[..., 1] -= mean[1]
    image[..., 2] -= mean[2]
    return image, label

def imgnt_mean_substract2(image,label):
    image /= 127.5
    image -= 1.
    return image,label

def imgnt_normalize_mean_substract(image,label):
    image = image / 255.0
    mean_tensor = -1.0 * tf.constant([0.485,0.456,0.406])
    std = tf.constant([0.229,0.224,0.225])
    
    if image.dtype != mean_tensor.dtype:
        image = tf.add(image, tf.cast(mean_tensor, image.dtype))
    else:
        image = tf.add(image, mean_tensor)
    
    image /= std
    
    return image


def RGBtoBGR_substractMeanRGBVal(image,label):
    """
    Useful for both VGG and Resnet 
    """
    mean = tf.constant([103.939, 116.779, 123.68])
    image = image[...,::-1] # convert RGB to BGR 
    mean_tensor = -1 * mean # mean tensor
    if image.dtype != mean_tensor.dtype:
        image = tf.add(image, tf.cast(mean_tensor, image.dtype))
    else:
        image = tf.add(image, mean_tensor)
    return image, label
 

def imgnt_preproc(train_ds,val_ds):
    train_ds = train_ds.map(imgnt_normalize_mean_substract)
    val_ds = val_ds.map(imgnt_normalize_mean_substract)
    return train_ds, val_ds

def cifar10_preproc(train_ds,test_ds):
    train_ds = train_ds.map(resize_img)
    test_ds = test_ds.map(resize_img)
    train_ds = train_ds.map(RGBtoBGR_substractMeanRGBVal)
    test_ds = test_ds.map(RGBtoBGR_substractMeanRGBVal)
    
    return train_ds, test_ds


def cifar100_preproc(train_ds,test_ds):
    train_ds = train_ds.map(resize_img)
    test_ds = test_ds.map(resize_img)
    train_ds = train_ds.map(normalize_image)
    test_ds = test_ds.map(normalize_image)
    return train_ds, test_ds
