import tensorflow as tf
import numpy as np
import time

class stop_acc_thresh(tf.keras.callbacks.Callback):
    """
    callback to stop training when a certain validation accuracy is reached
    """
    def __init__(self,acc):
        super(stop_acc_thresh,self).init__()
        self.acc_thresh = acc

    def on_epoch_end(self,epoch,logs={}):
        if (logs.get('val_accuracy') > self.acc_thresh):
            print("\n Reached %2.2f accuracy" %(self.acc_thresh*100))
            self.model.stop_training = True
        print('val acc = %2.2f' %(logs.get('val_acc')))
        
class measure_img_sec(tf.keras.callbacks.Callback):
    "Measure img_sec per batch and per epoch"""
    
    def __init__(self,batch_size):
        super(measure_img_sec,self).__init__()
        self.batch_size = batch_size
        self.per_epoch_log = []
        
    
    def on_epoch_begin(self,epoch,logs=None):
        self.per_batch_log = []
        
    
    def on_epoch_end(self,epoch,logs=None):
        curr_epoch_imgsec_mean = np.mean(np.array(self.per_batch_log))
        print('current epoch img/sec = ',curr_epoch_imgsec_mean)
        self.per_epoch_log.append(curr_epoch_imgsec_mean)
    
    def on_train_batch_begin(self,batch,logs=None):
        self.batch_time_start = time.time() 
         
    
    def on_train_batch_end(self,batch,logs=None):
        time_end = time.time()
        time_elapsed = time_end - self.batch_time_start
        
        img_sec = self.batch_size/time_elapsed
        #print('img_sec on current batch=',img_sec)
        self.per_batch_log.append(img_sec)


    def on_train_end(self,logs=None):
        """
        Print the avg img/sec at end of train 
        """
        overrall_img_sec = np.mean(np.array(self.per_epoch_log))
        print('overrall img_sec across all epochs=',overrall_img_sec)
        
        
def ResNetLRDecay(epoch,lr):
    if epoch == 30 or epoch == 60 or epoch == 80:
        return lr * 0.1
    else:
        return lr
    