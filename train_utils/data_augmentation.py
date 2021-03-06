import tensorflow as tf
from tensorflow import keras

# def do_rand_horizontal_flip(image,label):
#     image = tf.keras.layers.RandomFlip(mode='horizontal')(image)
#     return image, label

# def rand_horiz_flip(train_ds):
#     augment = tf.keras.Sequential()
#     train_ds = train_ds.map(do_rand_horizontal_flip, num_parallel_calls=tf.data.AUTOTUNE)
#     return train_ds

def rand_horiz_flip(train_ds):
    augment = tf.keras.Sequential()
    augment.add(tf.keras.layers.RandomFlip(mode='horizontal'))
    train_ds = train_ds.map(lambda x,y: (augment(x),y), num_parallel_calls=tf.data.AUTOTUNE)
    return train_ds



def imgnt_data_aug(train_ds):
    train_ds = rand_horiz_flip(train_ds)
    return train_ds

def cifar10_data_aug(train_ds):
    train_ds = rand_horiz_flip(train_ds)
    return train_ds

def cifar100_data_aug(train_ds):
    train_ds = rand_horiz_flip(train_ds)
    return train_ds
