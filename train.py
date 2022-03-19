# train the models from this entry point
# will be replaced
import tensorflow as tf

#from tensorflow.keras.applications.resnet50 import ResNet50

#from models.plain_resnets_imgnt import plain_resnet34

from models.resnet_imgnt import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

model = ResNet101()

model.summary()
