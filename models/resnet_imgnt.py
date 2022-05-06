import tensorflow as tf
from tensorflow import keras

def bottleneck_conv_block(input_tensor,filters,strides=(2,2)):
    """
    stride is 2x2 unless mentioned otherwise, only conv stage 2 has stride of 1,
     otherwise starting from stage 3 onwards the stride is of 2( which is default )

    """

    out = keras.layers.Conv2D(filters[0],(1,1),strides=strides, padding = 'valid',
                              kernel_initializer='he_normal',bias_initializer='he_normal',
                              kernel_regularizer= tf.keras.regularizers.L2(l2=1e-4),
                              bias_regularizer= tf.keras.regularizers.L2(l2=1e-4))(input_tensor)

    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation('relu')(out)


    out = tf.keras.layers.Conv2D(filters[1],(3,3),padding='same',
                                 kernel_initializer='he_normal',bias_initializer='he_normal',
                                kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4), bias_regularizer=tf.keras.regularizers.L2(l2=1e-4))(out)

    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation('relu')(out)

    out = tf.keras.layers.Conv2D(filters[2],(1,1),padding='valid',
                                 kernel_initializer='he_normal',bias_initializer='he_normal',
                                 kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4), bias_regularizer=tf.keras.regularizers.L2(l2=1e-4))(out)

    out = tf.keras.layers.BatchNormalization()(out)


    # let shortcut handling happen last

    # compute the shortcut
    shortcut = keras.layers.Conv2D(filters[2],(1,1),strides=strides,padding='same',kernel_initializer='he_normal',
                                   bias_initializer='he_normal',
                                   kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4), bias_regularizer=tf.keras.regularizers.L2(l2=1e-4))(input_tensor)
    shortcut = keras.layers.BatchNormalization()(shortcut)

    # add
    shortcut_add = tf.keras.layers.Add()([shortcut,out])

    # apply activation function
    final_out = tf.keras.layers.Activation('relu')(shortcut_add)

    return final_out

def bottleneck_identity_block(input_tensor,filters):


    out = keras.layers.Conv2D(filters[0], (1,1), strides=1, padding = 'valid',
                              kernel_initializer ='he_normal',bias_initializer = 'he_normal',
                              kernel_regularizer = tf.keras.regularizers.L2(l2=1e-4),
                              bias_regularizer = tf.keras.regularizers.L2(l2=1e-4))(input_tensor)

    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation('relu')(out)


    out = tf.keras.layers.Conv2D(filters[1],(3,3),padding='same',
                                 kernel_initializer='he_normal',bias_initializer='he_normal',
                                kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4), bias_regularizer=tf.keras.regularizers.L2(l2=1e-4))(out)

    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation('relu')(out)

    out = tf.keras.layers.Conv2D(filters[2],(1,1),padding='valid',
                                 kernel_initializer='he_normal',bias_initializer='he_normal',
                                 kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4), bias_regularizer=tf.keras.regularizers.L2(l2=1e-4))(out)

    out = tf.keras.layers.BatchNormalization()(out)

    # let shortcut handling happen last
    shortcut = input_tensor

    shortcut_add = tf.keras.layers.Add()([shortcut,out])

    final_out = tf.keras.layers.Activation('relu')(shortcut_add)

    return final_out




def ResNet50(input_shape=(224,224,3),num_classes=1000):

    in_tensor = tf.keras.Input(input_shape)

    #out = tf.keras.layers.ZeroPadding2D((3,3))(in_tensor)

    # conv1_x layer
    out = keras.layers.Conv2D(64,(7,7),strides=2,padding='same',
                                kernel_initializer='he_normal',bias_initializer='he_normal',
                                kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4), bias_regularizer=tf.keras.regularizers.L2(l2=1e-4))(in_tensor)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.Activation('relu')(out)


    # conv2_x layer
    out = keras.layers.MaxPool2D((3,3),strides=2,padding='same')(out)

    out = bottleneck_conv_block(out,[64,64,256],strides=(1,1))
    out = bottleneck_identity_block(out,[64,64,256])
    out = bottleneck_identity_block(out,[64,64,256])
    #print('out.shape=',out.shape)

    # conv3_x layer
    out = bottleneck_conv_block(out,[128,128,512],strides=(2,2))
    out = bottleneck_identity_block(out,[128,128,512])
    out = bottleneck_identity_block(out,[128,128,512])
    out = bottleneck_identity_block(out,[128,128,512])


    # conv4x layer
    out = bottleneck_conv_block(out,[256,256,1024],strides=(2,2))
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])

    # conv5x layer
    out = bottleneck_conv_block(out,[512,512,2048],strides=(2,2))
    out = bottleneck_identity_block(out,[512,512,2048])
    out = bottleneck_identity_block(out,[512,512,2048])

    out = keras.layers.GlobalAveragePooling2D()(out)
    out = keras.layers.Dense(1000,activation='softmax',
                            kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4), bias_regularizer=tf.keras.regularizers.L2(l2=1e-4))(out)

    return tf.keras.Model(inputs=in_tensor,outputs=out)



def ResNet101(input_shape=(224,224,3),num_classes=1000):

    in_tensor = tf.keras.Input(input_shape)

    #out = tf.keras.layers.ZeroPadding2D((3,3))(in_tensor)

    # conv1_x layer
    out = keras.layers.Conv2D(64,(7,7),strides=2,padding='same',kernel_initializer='he_normal',bias_initializer='he_normal')(in_tensor)

    # conv2_x layer
    out = keras.layers.MaxPool2D((3,3),strides=2,padding='same')(out)

    out = bottleneck_conv_block(out,[64,64,256],strides=(1,1))
    out = bottleneck_identity_block(out,[64,64,256])
    out = bottleneck_identity_block(out,[64,64,256])
    #print('out.shape=',out.shape)

    # conv3_x layer
    out = bottleneck_conv_block(out,[128,128,512],strides=(2,2))
    out = bottleneck_identity_block(out,[128,128,512])
    out = bottleneck_identity_block(out,[128,128,512])
    out = bottleneck_identity_block(out,[128,128,512])


    # conv4x layer
    out = bottleneck_conv_block(out,[256,256,1024],strides=(2,2))
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])

    # conv5x layer
    out = bottleneck_conv_block(out,[512,512,2048],strides=(2,2))
    out = bottleneck_identity_block(out,[512,512,2048])
    out = bottleneck_identity_block(out,[512,512,2048])



    out = keras.layers.GlobalAveragePooling2D()(out)
    out = keras.layers.Dense(1000,activation='softmax',
                            kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4), bias_regularizer=tf.keras.regularizers.L2(l2=1e-4))(out)

    return tf.keras.Model(inputs=in_tensor,outputs=out)


def ResNet152(input_shape=(224,224,3),num_classes=1000):

    in_tensor = tf.keras.Input(input_shape)
    # conv1_x layer
    out = keras.layers.Conv2D(64,(7,7),strides=2,padding='same',
                                kernel_initializer='he_normal',bias_initializer='he_normal',
                                kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4), bias_regularizer=tf.keras.regularizers.L2(l2=1e-4))(in_tensor)

    # conv2_x layer
    out = keras.layers.MaxPool2D((3,3),strides=2,padding='same')(out)
    out = bottleneck_conv_block(out,[64,64,256],strides=(1,1))
    out = bottleneck_identity_block(out,[64,64,256])
    out = bottleneck_identity_block(out,[64,64,256])
    #print('out.shape=',out.shape)

    # conv3_x layer
    out = bottleneck_conv_block(out,[128,128,512],strides=(2,2))
    out = bottleneck_identity_block(out,[128,128,512])
    out = bottleneck_identity_block(out,[128,128,512])
    out = bottleneck_identity_block(out,[128,128,512])
    out = bottleneck_identity_block(out,[128,128,512])
    out = bottleneck_identity_block(out,[128,128,512])
    out = bottleneck_identity_block(out,[128,128,512])
    out = bottleneck_identity_block(out,[128,128,512])


    # conv4x layer
    out = bottleneck_conv_block(out,[256,256,1024],strides=(2,2))
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])
    out = bottleneck_identity_block(out,[256,256,1024])

    # conv5x layer
    out = bottleneck_conv_block(out,[512,512,2048],strides=(2,2))
    out = bottleneck_identity_block(out,[512,512,2048])
    out = bottleneck_identity_block(out,[512,512,2048])


    out = keras.layers.GlobalAveragePooling2D()(out)
    out = keras.layers.Dense(1000,activation='softmax',
                            kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4), bias_regularizer=tf.keras.regularizers.L2(l2=1e-4))(out)

    return tf.keras.Model(inputs=in_tensor,outputs=out)



def BasicResidualBlock_identity_block(input_tensor,num_filters,strides=(2,2)):


    out = tf.keras.layers.Conv2D(filters[1],(3,3),padding='same',
                                 kernel_initializer='he_normal',bias_initializer='he_normal',
                                kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4), bias_regularizer=tf.keras.regularizers.L2(l2=1e-4))(out)

    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation('relu')(out)

    out = tf.keras.layers.Conv2D(filters[2],(3,3),padding='same',
                                 kernel_initializer='he_normal',bias_initializer='he_normal',
                                 kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4), bias_regularizer=tf.keras.regularizers.L2(l2=1e-4))(out)

    out = tf.keras.layers.BatchNormalization()(out)

    # let shortcut handling happen last
    shortcut = input_tensor

    shortcut_add = tf.keras.layers.Add()([shortcut,out])

    final_out = tf.keras.layers.Activation('relu')(shortcut_add)


    return final_out


def BasicResidualBlock_conv_block(input_tensor,num_filters,strides=(2,2)):


    out = tf.keras.layers.Conv2D(filters[0],(3,3),strides=strides,padding='same',
                                 kernel_initializer='he_normal',bias_initializer='he_normal',
                                kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4), bias_regularizer=tf.keras.regularizers.L2(l2=1e-4))(input_tensor)

    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation('relu')(out)

    out = tf.keras.layers.Conv2D(filters[1],(3,3),padding='same',
                                 kernel_initializer='he_normal',bias_initializer='he_normal',
                                 kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4), bias_regularizer=tf.keras.regularizers.L2(l2=1e-4))(out)

    out = tf.keras.layers.BatchNormalization()(out)

    # compute the shortcut
    shortcut = keras.layers.Conv2D(filters[1],(1,1),strides=strides,padding='same',kernel_initializer='he_normal',
                                   bias_initializer='he_normal',
                                   kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4), bias_regularizer=tf.keras.regularizers.L2(l2=1e-4))(input_tensor)
    shortcut = keras.layers.BatchNormalization()(shortcut)

    # add
    shortcut_add = tf.keras.layers.Add()([shortcut,out])

    # apply activation function
    final_out = tf.keras.layers.Activation('relu')(shortcut_add)

    return final_out





def ResNet18(input_shape=(224,224,3),num_classes=1000):

    in_tensor = keras.Input(input_shape)

    # conv1_x layer
    out = keras.layers.Conv2D(64,(7,7),strides=2,padding='same',
                                kernel_initializer='he_normal',bias_initializer='he_normal',
                                kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4), bias_regularizer=tf.keras.regularizers.L2(l2=1e-4))(in_tensor)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation('relu')(out)
    #print('conv1x layer output shape',out.shape)


    # conv2_x layer
    out = keras.layers.MaxPool2D((3,3),strides=2,padding='same')(out)
    out = BasicResidualBlock_conv_block(out,[64,64],strides=(1,1))
    out = BasicResidualBlock_identity_block(out,[64,64])

    # conv 3x layer
    out = BasicResidualBlock_conv_block(out,[128,128],strides=(2,2))
    out = BasicResidualBlock_identity_block(out,[128,128])

    # conv 4x layer
    out = BasicResidualBlock_conv_block(out,[256,256],strides=(2,2))
    out = BasicResidualBlock_identity_block(out,[256,256])

    # conv 5x layer
    out = BasicResidualBlock_conv_block(out,[512,512],strides=(2,2))
    out = BasicResidualBlock_identity_block(out,[512,512])


    out = keras.layers.GlobalAveragePooling2D()(out)
    out = keras.layers.Dense(1000,activation='softmax',
                            kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4), bias_regularizer=tf.keras.regularizers.L2(l2=1e-4))(out)
    #out = keras.layers.Activation('softmax')(out)

    return keras.Model(inputs=in_tensor,outputs=out)


def ResNet34(input_shape=(224,224,3),num_classes=1000):

    in_tensor = keras.Input(input_shape)

    # conv1_x layer
    out = keras.layers.Conv2D(64,(7,7),strides=2,padding='same',kernel_initializer='he_normal',bias_initializer='he_normal',
                             kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4), bias_regularizer=tf.keras.regularizers.L2(l2=1e-4))(in_tensor)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation('relu')(out)
    #print('conv1x layer output shape',out.shape)


    # conv2_x layer
    out = keras.layers.MaxPool2D((3,3),strides=2,padding='same')(out)
    out = BasicResidualBlock_conv_block(out,[64,64],strides=(1,1))
    out = BasicResidualBlock_identity_block(out,[64,64])
    out = BasicResidualBlock_identity_block(out,[64,64])

    # conv 3x layer
    out = BasicResidualBlock_conv_block(out,[128,128],strides=(2,2))
    out = BasicResidualBlock_identity_block(out,[128,128])
    out = BasicResidualBlock_identity_block(out,[128,128])
    out = BasicResidualBlock_identity_block(out,[128,128])


    # conv 4x layer
    out = BasicResidualBlock_conv_block(out,[256,256],strides=(2,2))
    out = BasicResidualBlock_identity_block(out,[256,256])
    out = BasicResidualBlock_identity_block(out,[256,256])
    out = BasicResidualBlock_identity_block(out,[256,256])
    out = BasicResidualBlock_identity_block(out,[256,256])
    out = BasicResidualBlock_identity_block(out,[256,256])

    # conv 5x layer
    out = BasicResidualBlock_conv_block(out,[512,512],strides=(2,2))
    out = BasicResidualBlock_identity_block(out,[512,512])
    out = BasicResidualBlock_identity_block(out,[512,512])


    out = keras.layers.GlobalAveragePooling2D()(out)
    out = keras.layers.Dense(1000,activation='softmax',
                            kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4), bias_regularizer=tf.keras.regularizers.L2(l2=1e-4))(out)
    
    return keras.Model(inputs=in_tensor,outputs=out)
