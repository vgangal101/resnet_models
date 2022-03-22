import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
import argparse
import math
import matplotlib.pyplot as plt
import scipy
import atexit
import time

# training relevant imports
from models.resnet_imgnt import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from train_utils.custom_callbacks import stop_acc_thresh, measure_img_sec
from train_utils.data_augmentation import imgnt_data_aug, cifar10_data_aug, cifar100_data_aug
from train_utils.preprocessing import imgnt_preproc, cifar10_preproc, cifar100_preproc

def get_args():
    parser = argparse.ArgumentParser(description='training configurations')
    parser.add_argument('--model',type=str,help='choices are ResNet18, ResNet34, ResNet50, ResNet101, ResNet152')
    parser.add_argument('--dataset',type=str,help='cifar10,cifar100,imagenet')
    parser.add_argument('--batch_size',type=int,default=256)
    # have the requirement that if the code is imagenet , then specify a path to dataset
    parser.add_argument('--imgnt_data_path',type=str,default='/data/petabyte/IMAGENET/Imagenet2012',help='only provide if imagenet is specified')
    parser.add_argument('--num_epochs',type=int,default=100,help='provide number of epochs to run')
    parser.add_argument('--lr',type=float,default=1e-2,help='learning rate to use')
    parser.add_argument('--momentum',type=float,default=0.9,help='value for momentum')
    parser.add_argument('--lr_schedule',type=str,default='constant',help='choice of learning rate scheduler')
    parser.add_argument('--lr_plat_patience',type=int,default=5,help='patience of epochs before reducing lr')
    parser.add_argument('--img_size',type=tuple, default=(224,224,3),help='imagenet crop size')
    parser.add_argument('--data_aug',type=bool,default=True,help='use data augmentation or not')
    parser.add_argument('--early_stopping', type=bool, default=False, help='use early stopping')
    parser.add_argument('--train_to_accuracy',type=float,default=0,help='using early stopping to train to certain percentage')
    parser.add_argument('--save_checkpoints',type=bool,default=False,help='whether to save checkpoints or not')
    parser.add_argument('--checkpoint_frequency',type=int,default=5,help='checkpointing frequency')
    parser.add_argument('--checkpoint_dir',type=str,default='./checkpoints',help='where to save checkpoints')
    parser.add_argument('--num_gpus',type=int,default=1,help='number of gpus to use (on node)')
    parser.add_argument('--measure_img_sec',type=bool,default=False,help='measure img/sec')
    parser.add_argument('--resume_training',type=bool,default=False,help='resume_training')
    args = parser.parse_args()
    return args

def get_imagenet_dataset(args):
    print('loading imagenet dataset')
    path_dir = args.imgnt_data_path

    if 'train' in path_dir or 'val' in path_dir :
        raise ValueError('Specify the root directory not the train directory for the imagenet dataset')

    path_train = path_dir + '/train'
    path_val = path_dir + '/val'

    IMG_SIZE = args.img_size[:2]

    train_dataset = tf.keras.utils.image_dataset_from_directory(path_train,image_size=IMG_SIZE,batch_size=args.batch_size)
    val_dataset = tf.keras.utils.image_dataset_from_directory(path_val,image_size=IMG_SIZE, batch_size=args.batch_size)



    return train_dataset, val_dataset



def preprocess_dataset(args,train_dataset,test_dataset):
    """
    train_dataset : tf.data.Dataset
    test_dataset : tf.data.Dataset

    should return normalized image data + any data augmentation as needed.
    """
    if args.dataset == 'cifar10':
        return cifar10_preproc(train_dataset,test_dataset)
    elif args.dataset == 'cifar100':
        return cifar100_preproc(train_dataset,test_dataset)
    elif args.dataset == 'imagenet':
        return imgnt_preproc(train_dataset,test_dataset)




def get_dataset(args):
    # should return a TF dataset object for train, test
    train_dataset = None
    test_dataset = None

    dataset_name = args.dataset

    if dataset_name.lower() == 'cifar10':
        (x_train,y_train), (x_test,y_test) = tf.keras.datasets.cifar10.load_data()
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
        return train_dataset.batch(args.batch_size), test_dataset.batch(args.batch_size)
    elif dataset_name.lower() == 'cifar100':
        (x_train,y_train), (x_test,y_test) = tf.keras.datasets.cifar100.load_data()
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
        return train_dataset.batch(args.batch_size), test_dataset.batch(args.batch_size)
    elif dataset_name.lower() == 'imagenet':
        return get_imagenet_dataset(args)



def plot_training(history,args):
    print('history.history=',history.history)
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    val_top1_acc = history.history['val_top1_acc']
    val_top5_acc = history.history['val_top5_acc']

    plt.figure()
    plt.title("Epoch vs Accuracy")
    plt.plot(accuracy,label='training accuracy')
    plt.plot(val_accuracy,label='val_accuracy')
    plt.plot(val_top1_acc,label='val_top1_acc')
    plt.plot(val_top5_acc,label='val_top5_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')


    viz_file = 'accuracy_graph_' + args.dataset.lower() + '_' + args.model.lower() + '_bs' + str(args.batch_size) + '_epochs' + str(args.num_epochs) + '.png'
    plt.savefig(viz_file)
    plt.show()


    plt.figure()
    plt.title("Epoch vs Loss")
    plt.plot(history.history['loss'],label='training loss')
    plt.plot(history.history['val_loss'],label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')

    viz_file2 = 'loss_graph_' + args.dataset.lower() + '_' + args.model.lower() + '_bs' + str(args.batch_size) + '_epochs' + str(args.num_epochs) + '.png'
    plt.savefig(viz_file2)
    plt.show()

def get_model(args, num_classes, img_shape):
    model = None
    if args.model.lower() == 'paper_vgg11':
        model = paper_models.VGG11_A(num_classes,img_shape)
    elif args.model.lower() == 'paper_vgg13':
        model = paper_models.VGG13_B(num_classes,img_shape)
    elif args.model.lower() == 'paper_vgg16c':
        model = paper_models.VGG16_C(num_classes,img_shape)
    elif args.model.lower() == 'paper_vgg16d':
        model = paper_models.VGG16_D(num_classes,img_shape)
    elif args.model.lower() == 'paper_vgg19e':
        model = paper_models.VGG19_E(num_classes,img_shape)
    elif args.model.lower() == 'bn_vgg16':
        model = bn_vgg.bn_VGG16D(num_classes,img_shape)
    elif args.model.lower() == 'bn_vgg19':
        model = bn_vgg.bn_VGG19E(num_classes,img_shape)
    else:
        raise ValueError('Invalid value for the model name' + 'got model name= ' + args.model)

    return model


def get_dataset_props(args):
    num_classes = None
    img_shape = None
    dataset_name = args.dataset

    if dataset_name == None:
        raise ValueError('No dataset specified. Exiting')
    elif dataset_name.lower() == 'cifar10':
        img_shape = (32,32,3)
        num_classes = 10
    elif dataset_name.lower() == 'cifar100':
        img_shape = (32,32,3)
        num_classes = 100
    elif dataset_name.lower() == 'imagenet':
        img_shape = (224,224,3)
        num_classes = 1000
    else:
        raise ValueError('Invalid dataset specified, dataset specified=', args.dataset)

    return num_classes,img_shape


def get_callbacks_and_optimizer(args):
    callbacks = []
    optimizer = None
    momentum = args.momentum



    if args.lr_schedule == 'constant':
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr,momentum=momentum)
    elif args.lr_schedule == 'time':
        decay = args.lr / args.num_epochs
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr,momentum=momentum)

        def lr_time_based_decay(epoch,lr):
            return lr * 1 / (1 + decay * epoch)
        lr_callback = LearningRateScheduler(lr_time_based_decay,verbose=1)

        callbacks.append(lr_callback)

    elif args.lr_schedule == 'step_decay':
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr,momentum=momentum)
        initial_learning_rate = args.lr

        def lr_step_decay(epoch,lr):
            drop_rate = 0.5
            epochs_drop = 10.0
            return initial_learning_rate * math.pow(drop_rate, math.floor(epoch/epochs_drop))

        lr_callback = LearningRateScheduler(lr_step_decay,verbose=1)
        callbacks.append(lr_callback)

    elif args.lr_schedule == 'exp_decay':
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr,momentum=momentum)
        initial_learning_rate = args.lr

        def lr_exp_decay(epoch,lr):
            k = 0.1
            return initial_learning_rate * math.exp(-k*epoch)

        lr_callback = LearningRateScheduler(lr_exp_decay,verbose=1)
        callbacks.append(lr_callback)

    else:
        raise ValueError('invalid value for learning rate scheduler got: ', args.lr_scheduler)

    #ReduceLROnPlateau callback
    reduce_lr_plat = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',factor=0.1,patience=args.lr_plat_patience)
    callbacks.append(reduce_lr_plat)

    if args.train_to_accuracy != 0:
        cb = stop_acc_thresh(args.train_to_accuracy)
        callbacks.append(cb)

    # early stopping
    if args.early_stopping:
        ea = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=10,verbose=2)
        callbacks.append(ea)

    if args.save_checkpoints:
        #checkpoint_path = args.checkpoint_dir + 'cp-{epoch:04d}.ckpt'
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=args.checkpoint_dir,monitor='val_accuracy',save_freq='epoch',verbose=1)
        callbacks.append(cp_callback)

    if args.measure_img_sec:
        callbacks.append(measure_img_sec(args.batch_size))

    return callbacks, optimizer



def apply_data_aug(args,train_ds):
    """
    Takes in a tf.data.dataset, applies data augmentation
    """


    if args.dataset == 'imagenet':
        train_ds = imgnt_data_aug(train_ds)
    elif args.dataset == 'cifar10':
        train_ds = cifar10_data_aug(train_ds)
    elif args.dataset == 'cifar100':
        train_ds = cifar100_data_aug(train_ds)

    return train_ds


def main():

    args = get_args()

    num_classes, img_shape = get_dataset_props(args)

    model = get_model(args,num_classes,img_shape)

    print("preparing data")

    train_dataset, test_dataset = get_dataset(args)

    train_dataset, test_dataset = preprocess_dataset(args,train_dataset,test_dataset)

    if args.data_aug == True:
        train_dataset = apply_data_aug(args,train_dataset)

    callbacks, optimizer = get_callbacks_and_optimizer(args)

    BUFFER_SIZE = 10000
    #train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE).cache().shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    print('data preparation complete')

    print('setting up distribution strategy')
    gpus = tf.config.list_logical_devices('GPU')
    print('number of gpus used = ',args.num_gpus)
    strategy = tf.distribute.MirroredStrategy(gpus[:args.num_gpus])


    print('preparing model')
    with strategy.scope():

        # MODEL LOADING AND RESTORE CODE SEEMS TO FIT HERE !!
        model = None

        if args.resume_training:
            loaded_model = tf.keras.models.load_model(args.checkpoint_dir)
        else:
            model = get_model(args,num_classes,img_shape)
            print('model is ready, model chosen=',args.model.lower())

            print('compiling model with essential necessities ....')
            model.compile(optimizer=optimizer,
                        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy',tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1,name='top1_acc'),tf.keras.metrics.TopKCategoricalAccuracy(k=5,name='top5_acc')])




    print("starting training")
    #time_start = time.time()
    history = model.fit(train_dataset,epochs=args.num_epochs,validation_data=test_dataset,callbacks=callbacks)
    #time_end = time.time()

    #print('time to complete training',time_end-time_start)
    #print('history.history.keys()=',history.history.keys())
    print('training complete')

    print('plotting...')
    plot_training(history,args)
    print('plotting complete')

    test_loss, test_acc = model.evaluate(test_dataset)
    print("test_loss=",test_loss)
    print("test_acc",test_acc)

    train_eval_log_file = open('./train_eval_file_' + args.model + '_' +  args.dataset + '_' + str(args.batch_size) + '.log', 'w')

    train_eval_log_file.write("test results\n")

    train_eval_log_file.write('test_loss' + '=' + str(test_loss) + '\n')
    train_eval_log_file.write('test_acc' + '=' + str(test_acc) + '\n')

    save_to_dir = args.model.lower() + '_' + args.dataset.lower() + '_' +  str(args.batch_size)
    model.save(save_to_dir)
    print("training and eval complete")
    atexit.register(strategy._extended._collective_ops._pool.close)


if __name__ == '__main__':
    main()
