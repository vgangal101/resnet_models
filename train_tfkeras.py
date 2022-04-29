import sys
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
import tensorflow.keras.backend as K
import pathlib
import os

# training relevant imports

from train_utils.custom_callbacks import stop_acc_thresh, measure_img_sec, ResNetLRDecay
from train_utils.data_augmentation import imgnt_data_aug, cifar10_data_aug, cifar100_data_aug
from train_utils.preprocessing import imgnt_preproc, cifar10_preproc, cifar100_preproc

# model imports 
from models.resnet_imgnt import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

#tf.debugging.set_log_device_placement(True)

def get_args():
    parser = argparse.ArgumentParser(description='training configurations')
    parser.add_argument('--model',type=str,help='choices are ResNet18, ResNet34, ResNet50, ResNet101, ResNet152') # either vgg11,13,16,19 , now contains batch normalized options as well
    parser.add_argument('--dataset',type=str,help='cifar10,cifar100,imagenet')
    parser.add_argument('--batch_size',type=int,default=256)
    # have the requirement that if the code is imagenet , then specify a path to dataset
    parser.add_argument('--imgnt_data_path',type=str,default='/data/petabyte/IMAGENET/Imagenet2012',help='only provide if imagenet is specified')
    parser.add_argument('--imgnt_labels_file',type=str,default='./imagenet_labels.txt')
    parser.add_argument('--num_epochs',type=int,default=100,help='provide number of epochs to run')
    parser.add_argument('--lr',type=float,default=1e-2,help='learning rate to use')
    parser.add_argument('--momentum',type=float,default=0.9,help='value for momentum')
    parser.add_argument('--lr_schedule',type=str,default='constant',help='choice of learning rate scheduler')
    parser.add_argument('--img_size',type=tuple, default=(224,224,3),help='imagenet crop size')
    parser.add_argument('--data_aug',type=bool,default=True,help='use data augmentation or not')
    parser.add_argument('--early_stopping', type=bool, default=False, help='use early stopping')
    parser.add_argument('--train_to_accuracy',type=float,default=0,help='using early stopping to train to certain percentage')
    parser.add_argument('--save_checkpoints',type=bool,default=False,help='whether to save checkpoints or not')
    parser.add_argument('--checkpoint_frequency',type=int,default=5,help='checkpointing frequency')
    parser.add_argument('--checkpoint_dir',type=str,default='./checkpoints',help='where to save checkpoints')
    parser.add_argument('--reload_checkpoint',type=str,help='checkpoint to resume training from')
    parser.add_argument('--num_gpus',type=int,default=1,help='number of gpus to use (on node)')
    parser.add_argument('--measure_img_sec',type=bool,default=False,help='measure img/sec')
    parser.add_argument('--resume_training',type=bool,default=False,help='resume_training')
    parser.add_argument('--start_epoch',type=int,default=0,help='epoch to start at')
    parser.add_argument('--custom_lr_schedule',type=bool,default=False,help='use a custom lr schedule or not')
    parser.add_argument('--reduce_lr_on_plateau',type=bool,default=False,help='use the ReduceLrOnPlateau callback')
    parser.add_argument('--lr_plat_patience',type=int,default=5,help='patience of epochs before reducing lr, use it with callback')
    parser.add_argument('--min_lr',type=float,default=1e-4,help='lower bound on lr for ReduceLROnPlateau')
    parser.add_argument('--decay_epochs',type=float,default=30,help='how many epochs for each decay')
    parser.add_argument('--decay_rate',type=float,default=0.1,help='decay rate to use')
    args = parser.parse_args()
    return args
    

def get_imagenet_labels_classes(args):
    label_map = {}
    labels_file = args.imgnt_labels_file
    
    with open(labels_file,'r') as f:
        for l in f.readlines():
            proc_l = l.strip()
            data = proc_l.split(' ')
            imgnt_class = data[0]
            real_class = ' '.join(data[1:])
            label_map[imgnt_class] = real_class
        
    labels_list = list(label_map.values())
    
    return label_map, labels_list


def get_imagenet_dataset(args):
    
    labels_map, labels_list = get_imagenet_labels_classes(args)
    
    base_dir = args.imgnt_data_path
    
    train_dir = os.path.join(base_dir,'train')
    val_dir = os.path.join(base_dir,'val')
    
    print('Loading training dataset from memory')
    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,batch_size=args.batch_size,label_mode='categorical',image_size=(256,256))
    print('Loaded Training dataset')
    val_dataset = tf.keras.utils.image_dataset_from_directory(val_dir,batch_size=args.batch_size,label_mode='categorical',image_size=(256,256))
    
    # do random cropping here 
    
    def random_crop_op(img,label):
        return tf.keras.layers.RandomCrop(224,224)(img), label
    
    def center_crop(img,label):
        return tf.keras.layers.CenterCrop(224,224)(img), label
    
    
    print("doing Random Cropping on train_dataset")
    train_dataset = train_dataset.map(random_crop_op, num_parallel_calls=tf.data.AUTOTUNE)
    print('Train_dataset random cropping complete')
    
    print('doing center crops on val dataset')
    val_dataset = val_dataset.map(center_crop, num_parallel_calls=tf.data.AUTOTUNE)
    print('Val Dataset Center Cropping complete')
    
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
    
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    top5_acc = history.history['top5_acc']
    plt.figure()
    plt.title("Epoch vs Accuracy")
    plt.plot(accuracy,label='training accuracy')
    plt.plot(val_accuracy,label='val_accuracy')
    plt.plot(top5_acc,label='top5 accuracy')
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

    if args.model == 'ResNet18':
        model = ResNet18(input_shape=img_shape,num_classes=num_classes)
    elif args.model == 'ResNet34':
        model = ResNet34(input_shape=img_shape,num_classes=num_classes)
    elif args.model == 'ResNet50':
        model = ResNet50(input_shape=img_shape,num_classes=num_classes)
    elif args.model == 'ResNet101':
        model = ResNet101(input_shape=img_shape,num_classes=num_classes)
    elif args.model == 'ResNet152':
        model = ResNet152(input_shape=img_shape,num_classes=num_classes)
    else:
        raise ValueError("Invalid model name got: ",args.model)

    return model


def get_dataset_props(args):
    num_classes = None
    img_shape = None
    dataset_name = args.dataset

    if dataset_name == None:
        raise ValueError('No dataset specified. Exiting')
    elif dataset_name.lower() == 'cifar10':
        img_shape = (224,224,3)
        num_classes = 10
    elif dataset_name.lower() == 'cifar100':
        img_shape = (224,224,3)
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
        optimizer = tf.keras.optimizers.SGD(
          tf.keras.optimizers.schedules.ExponentialDecay(args.lr, decay_steps = args.decay_epochs * 1281167 / args.batch_size, decay_rate = args.decay_rate), 
          momentum = args.momentum);
    else:
        raise ValueError('invalid value for learning rate scheduler got: ', args.lr_scheduler)


    if args.reduce_lr_on_plateau:
        reduce_lr_plat = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',factor=0.1,patience=args.lr_plat_patience,min_lr=args.min_lr)
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

    if args.custom_lr_schedule == True:
        schedule = tf.keras.callbacks.LearningRateScheduler(ResNetLRDecay)
        callbacks.append(schedule)

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
    
    cmd_store = open("store_cmd.txt",'w')
    cmd_store.write(str(sys.argv[1:]))
    cmd_store.close()

    
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
        if args.resume_training:
            if args.reload_checkpoint == None:
                raise RuntimeError('no checkpoint specified')

            model = tf.keras.models.load_model(args.reload_checkpoint)
            print('updating the learning rate, old lr = ', tf.keras.backend.get_value(model.optimizer.lr))
            K.set_value(model.optimizer.lr,args.lr)
            print('new lr',args.lr)
        else:
            # MODEL LOADING AND RESTORE CODE SEEMS TO FIT HERE !!
            model = get_model(args,num_classes,img_shape)
            print('model is ready, model chosen=',args.model.lower())

            print('compiling model with essential necessities ....')
            model.compile(optimizer=optimizer,
                      loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy',tf.keras.metrics.TopKCategoricalAccuracy(k=1,name='top1_acc'),tf.keras.metrics.TopKCategoricalAccuracy(k=5,name='top5_acc')])

    print("starting training")
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    #time_start = time.time()
    history = model.fit(train_dataset,initial_epoch=args.start_epoch,epochs=args.num_epochs,validation_data=test_dataset,callbacks=callbacks)
    #time_end = time.time()

    #print('time to complete training',time_end-time_start)
    #print('history.history.keys()=',history.history.keys())
    print('training complete')

    print('plotting...')
    plot_training(history,args)
    print('plotting complete')

    save_to_dir = args.model.lower() + '_' + args.dataset.lower() + '_' + 'bs' + str(args.batch_size) + 'epochs' + str(args.num_epochs)
    model.save(save_to_dir)

    test_loss, test_acc, top1_acc, top5_acc = model.evaluate(test_dataset)
    print("test_loss=",test_loss)
    print("test_acc=",test_acc)
    print('top1_acc=',top1_acc)
    print('top5_acc=',top5_acc)

    train_eval_log_file = open('./train_eval_file_' + args.model + '_' +  args.dataset + '_' + str(args.batch_size) + 'epochs' + str(args.num_epochs) +  '.log', 'w')

    train_eval_log_file.write("test results\n")

    train_eval_log_file.write('test_loss' + '=' + str(test_loss) + '\n')
    train_eval_log_file.write('test_acc' + '=' + str(test_acc) + '\n')
    train_eval_log_file.write('top1_acc' + '=' + str(top1_acc) + '\n')
    train_eval_log_file.write('top5_acc' + '=' + str(top5_acc) + '\n')

    print("training and eval complete")
    atexit.register(strategy._extended._collective_ops._pool.close)


if __name__ == '__main__':
    main()
