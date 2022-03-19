import tensorflow as tf
from tensorflow import keras

def conv_block_plain_model_18_34(model,num_filters,downsample=False):
    if downsample == True:
        model.add(keras.layers.Conv2D(num_filters,(3,3),strides=2,padding='same'))
    else:
        model.add(keras.layers.Conv2D(num_filters,(3,3),padding='same'))

    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(num_filters,(3,3),padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))



def plain_resnet34(img_shape=(224,224,3),num_classes=1000):

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=img_shape))
    model.add(keras.layers.Conv2D(64,(7,7),padding='same',strides=(2,2)))
    model.add(keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2)))

    # conv layer conv2_x
    conv_block_plain_model_18_34(model,64,downsample=False)
    conv_block_plain_model_18_34(model,64,downsample=False)

    #model.summary()
    # conv layer conv3_x
    conv_block_plain_model_18_34(model,128,downsample=True)
    conv_block_plain_model_18_34(model,128,downsample=False)

    # conv layer conv4_x
    conv_block_plain_model_18_34(model,256,downsample=True)
    conv_block_plain_model_18_34(model,256,downsample=False)

    # conv layer conv5_x
    conv_block_plain_model_18_34(model,512,downsample=True)
    conv_block_plain_model_18_34(model,512,downsample=False)

    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(num_classes))
    model.add(keras.layers.Activation('relu'))

    #model.summary()

    return model



def plain_resnet18():
    img_shape = (224,224,3)
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=img_shape))
    model.add(keras.layers.Conv2D(64,(7,7),strides=(2,2)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2)))

    model.add(keras.layers.Conv2D(64,(3,3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(64,(3,3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Conv2D(128,(3,3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(128,(3,3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))


    model.add(keras.layers.Conv2D(256,(3,3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(256,(3,3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    # write out plain resnet 34 first , this will then make more sense


    return model
