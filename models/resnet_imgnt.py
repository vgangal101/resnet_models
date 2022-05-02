import tensorflow as tf
from tensorflow import keras

def conv_block(input,filters,downsample=False):
    if downsample == True :
        out = keras.layers.Conv2D(filters[0],(1,1), strides = 2, padding = 'same',kernel_initializer='he_normal',bias_initializer='he_normal',kernel_regularizer=keras.regularizers.L2(0.0001),bias_regularizer=keras.regularizers.L2(0.0001))(input)
        shortcut = keras.layers.Conv2D(filters[2],(1,1),strides=2,padding='same',kernel_initializer='he_normal',bias_initializer='he_normal',kernel_regularizer=keras.regularizers.L2(0.0001),bias_regularizer=keras.regularizers.L2(0.0001))(input)
        shortcut = keras.layers.BatchNormalization()(shortcut)
    else:

        out = keras.layers.Conv2D(filters[0],(1,1),padding = 'same',kernel_initializer='he_normal',bias_initializer='he_normal',kernel_regularizer=keras.regularizers.L2(0.0001),bias_regularizer=keras.regularizers.L2(0.0001))(input)
        shortcut = keras.layers.Conv2D(filters[2],(1,1),strides=1,padding='same',kernel_initializer='he_normal',bias_initializer='he_normal',kernel_regularizer=keras.regularizers.L2(0.0001),bias_regularizer=keras.regularizers.L2(0.0001))(input)
        shortcut = keras.layers.BatchNormalization()(shortcut)


    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation('relu')(out)

    out = tf.keras.layers.Conv2D(filters[1],(3,3),padding='same',kernel_initializer='he_normal',bias_initializer='he_normal',kernel_regularizer=keras.regularizers.L2(0.0001),bias_regularizer=keras.regularizers.L2(0.0001))(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation('relu')(out)

    out = tf.keras.layers.Conv2D(filters[2],(1,1),padding='same',kernel_initializer='he_normal',bias_initializer='he_normal',kernel_regularizer=keras.regularizers.L2(0.0001),bias_regularizer=keras.regularizers.L2(0.0001))(out)
    out = tf.keras.layers.BatchNormalization()(out)

    short_cut_add = tf.keras.layers.Add()([shortcut,out])
    out = tf.keras.layers.Activation('relu')(out)

    return short_cut_add


def ResNet50(input_shape=(224,224,3),num_classes=1000):

    in_tensor = tf.keras.Input(input_shape)

    #out = tf.keras.layers.ZeroPadding2D((3,3))(in_tensor)

    # conv1_x layer
    out = keras.layers.Conv2D(64,(7,7),strides=2,padding='same',kernel_initializer='he_normal',bias_initializer='he_normal',kernel_regularizer=keras.regularizers.L2(0.0001),bias_regularizer=keras.regularizers.L2(0.0001))(in_tensor)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.Activation('relu')(out)


    # conv2_x layer
    out = keras.layers.MaxPool2D((3,3),strides=2,padding='same')(out)
    out = conv_block(out,[64,64,256],downsample=False)
    out = conv_block(out,[64,64,256],downsample=False)
    out = conv_block(out,[64,64,256],downsample=False)
    #print('out.shape=',out.shape)

    # conv3_x layer
    out = conv_block(out,[128,128,512],downsample=True)
    out = conv_block(out,[128,128,512],downsample=False)
    out = conv_block(out,[128,128,512],downsample=False)
    out = conv_block(out,[128,128,512],downsample=False)

    # conv4_x layer
    out = conv_block(out,[256,256,1024],downsample=True)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)

    #conv5_x layer
    out = conv_block(out,[512,512,2048],downsample=True)
    out = conv_block(out,[512,512,2048],downsample=False)
    out = conv_block(out,[512,512,2048],downsample=False)


    out = keras.layers.GlobalAveragePooling2D()(out)
    out = keras.layers.Dense(1000)(out)

    return tf.keras.Model(inputs=in_tensor,outputs=out)



def ResNet101(input_shape=(224,224,3),num_classes=1000):

    in_tensor = tf.keras.Input(input_shape)

    #out = tf.keras.layers.ZeroPadding2D((3,3))(in_tensor)

    # conv1_x layer
    out = keras.layers.Conv2D(64,(7,7),strides=2,padding='same',kernel_initializer='he_normal',bias_initializer='he_normal',kernel_regularizer=keras.regularizers.L2(0.0001),bias_regularizer=keras.regularizers.L2(0.0001))(in_tensor)

    # conv2_x layer
    out = keras.layers.MaxPool2D((3,3),strides=2,padding='same')(out)
    out = conv_block(out,[64,64,256],downsample=True)
    out = conv_block(out,[64,64,256],downsample=False)
    out = conv_block(out,[64,64,256],downsample=False)

    # conv3_x layer
    out = conv_block(out,[128,128,512],downsample=True)
    out = conv_block(out,[128,128,512],downsample=False)
    out = conv_block(out,[128,128,512],downsample=False)
    out = conv_block(out,[128,128,512],downsample=False)

    # conv4_x layer
    out = conv_block(out,[256,256,1024],downsample=True)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)

    #conv5_x layer
    out = conv_block(out,[512,512,2048],downsample=True)
    out = conv_block(out,[512,512,2048],downsample=False)
    out = conv_block(out,[512,512,2048],downsample=False)


    out = keras.layers.GlobalAveragePooling2D()(out)
    out = keras.layers.Dense(1000)(out)

    return tf.keras.Model(inputs=in_tensor,outputs=out)


def ResNet152(input_shape=(224,224,3),num_classes=1000):

    in_tensor = tf.keras.Input(input_shape)

    # conv1_x layer
    out = keras.layers.Conv2D(64,(7,7),strides=2,padding='same',kernel_initializer='he_normal',bias_initializer='he_normal',kernel_regularizer=keras.regularizers.L2(0.0001),bias_regularizer=keras.regularizers.L2(0.0001))(in_tensor)

    # conv2_x layer
    out = keras.layers.MaxPool2D((3,3),strides=2,padding='same')(out)
    out = conv_block(out,[64,64,256],downsample=False)
    out = conv_block(out,[64,64,256],downsample=False)
    out = conv_block(out,[64,64,256],downsample=False)

    # conv3_x layer
    out = conv_block(out,[128,128,512],downsample=True)
    out = conv_block(out,[128,128,512],downsample=False)
    out = conv_block(out,[128,128,512],downsample=False)
    out = conv_block(out,[128,128,512],downsample=False)
    out = conv_block(out,[128,128,512],downsample=False)
    out = conv_block(out,[128,128,512],downsample=False)
    out = conv_block(out,[128,128,512],downsample=False)
    out = conv_block(out,[128,128,512],downsample=False)


    # conv4_x layer
    out = conv_block(out,[256,256,1024],downsample=True)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)
    out = conv_block(out,[256,256,1024],downsample=False)

    #conv5_x layer
    out = conv_block(out,[512,512,2048],downsample=True)
    out = conv_block(out,[512,512,2048],downsample=False)
    out = conv_block(out,[512,512,2048],downsample=False)


    out = keras.layers.GlobalAveragePooling2D()(out)
    out = keras.layers.Dense(1000)(out)

    return tf.keras.Model(inputs=in_tensor,outputs=out)





def identity_residual_block(input,num_filters,downsample=False):
    if downsample == True :
        x = keras.layers.Conv2D(num_filters,(3,3), strides = 2, padding = 'same',kernel_initializer='he_normal',bias_initializer='he_normal',kernel_regularizer=keras.regularizers.L2(0.0001),bias_regularizer=keras.regularizers.L2(0.0001))(input)
        shortcut = keras.layers.Conv2D(num_filters,(1,1),strides=2,padding='same',kernel_initializer='he_normal',bias_initializer='he_normal',kernel_regularizer=keras.regularizers.L2(0.0001),bias_regularizer=keras.regularizers.L2(0.0001))(input)
        shortcut = keras.layers.BatchNormalization()(shortcut)
    else:
        x = keras.layers.Conv2D(num_filters,(3,3),padding = 'same',kernel_initializer='he_normal',bias_initializer='he_normal',kernel_regularizer=keras.regularizers.L2(0.0001),bias_regularizer=keras.regularizers.L2(0.0001))(input)
        shortcut = input

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(num_filters,(3,3),padding = 'same',kernel_initializer='he_normal',bias_initializer='he_normal',kernel_regularizer=keras.regularizers.L2(0.0001),bias_regularizer=keras.regularizers.L2(0.0001))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Add()([x,shortcut])
    x = keras.layers.Activation('relu')(x)

    return x


def ResNet18(input_shape=(224,224,3),num_classes=1000):

    in_tensor = keras.Input(input_shape)

    # conv1_x layer
    out = keras.layers.Conv2D(64,(7,7),strides=2,padding='same',kernel_initializer='he_normal',bias_initializer='he_normal',kernel_regularizer=keras.regularizers.L2(0.0001),bias_regularizer=keras.regularizers.L2(0.0001))(in_tensor)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation('relu')(out)
    #print('conv1x layer output shape',out.shape)


    # conv2_x layer
    out = keras.layers.MaxPool2D((3,3),strides=2,padding='same')(out)
    out = identity_residual_block(out,64,downsample=False)
    out = identity_residual_block(out,64,downsample=False)
    #print('conv2x layer output shape',out.shape)

    # conv3_x layer
    out = identity_residual_block(out,128,downsample=True)
    out = identity_residual_block(out,128,downsample=False)
    #print('conv3x layer output shape',out.shape)

    # conv4_x layer
    out = identity_residual_block(out,256,downsample=True)
    out = identity_residual_block(out,256,downsample=False)
    #print('conv4x layer output shape',out.shape)

    # conv5_x layer
    out = identity_residual_block(out,512,downsample=True)
    out = identity_residual_block(out,512,downsample=False)
    #print('conv4x layer output shape',out.shape)


    out = keras.layers.GlobalAveragePooling2D()(out)
    out = keras.layers.Dense(1000)(out)
    #out = keras.layers.Activation('softmax')(out)

    return keras.Model(inputs=in_tensor,outputs=out)


def ResNet34(input_shape=(224,224,3),num_classes=1000):

    in_tensor = keras.Input(input_shape)

    # conv1_x layer
    out = keras.layers.Conv2D(64,(7,7),strides=2,padding='same',kernel_initializer='he_normal',bias_initializer='he_normal',kernel_regularizer=keras.regularizers.L2(0.0001),bias_regularizer=keras.regularizers.L2(0.0001))(in_tensor)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation('relu')(out)
    #print('conv1x layer output shape',out.shape)


    # conv2_x layer
    out = keras.layers.MaxPool2D((3,3),strides=2,padding='same')(out)
    out = identity_residual_block(out,64,downsample=False) # erroring out here received shapes (28,28,64) and (55,55,64)
    out = identity_residual_block(out,64,downsample=False)
    #print('conv2x layer output shape',out.shape)


    # conv3_x layer
    out = identity_residual_block(out,128,downsample=True)
    out = identity_residual_block(out,128,downsample=False)
    out = identity_residual_block(out,128,downsample=False)
    out = identity_residual_block(out,128,downsample=False)
    #print('conv3x layer output shape',out.shape)

    # conv4_x layer
    out = identity_residual_block(out,256,downsample=True)
    out = identity_residual_block(out,256,downsample=False)
    out = identity_residual_block(out,256,downsample=False)
    out = identity_residual_block(out,256,downsample=False)
    #print('conv4x layer output shape',out.shape)

    # conv5_x layer
    out = identity_residual_block(out,512,downsample=True)
    out = identity_residual_block(out,512,downsample=False)
    out = identity_residual_block(out,512,downsample=False)
    #print('conv5x layer output shape',out.shape)

    out = keras.layers.GlobalAveragePooling2D()(out)

    out = keras.layers.Dense(1000)(out)

    return keras.Model(inputs=in_tensor,outputs=out)
