


Given Model Accuracy - Accuracy on test data is: 82.48

Model Definition
--

from keras.layers.convolutional import DepthwiseConv2D, SeparableConv2D, Conv2D
from keras.layers import Activation, GlobalAveragePooling2D, AveragePooling2D

   

model_new = Sequential()

model_new.add(SeparableConv2D(64, 3, input_shape=(32, 32, 3),border_mode='same', use_bias=False, name = "Block1")) # RF 3X3
model_new.add(Activation('relu'))
model_new.add(BatchNormalization())
model_new.add(Dropout(0.05))
# 32x32x64 


model_new.add(SeparableConv2D(64, 3 ,activation='relu',border_mode='valid', use_bias=False)) # RF 5X5
model_new.add(BatchNormalization())
model_new.add(Dropout(0.05))
# 28x28x64

model_new.add(SeparableConv2D(128, 3, activation='relu',border_mode='same', use_bias=False)) # RF 7X7
model_new.add(BatchNormalization())
model_new.add(Dropout(0.05))
# 26x26x128

print("Block1-------------------------------------------------------------------------------------------")

model_new.add(SeparableConv2D(64, 3 ,activation='relu',border_mode='valid', use_bias=False)) # RF 9X9
model_new.add(BatchNormalization())
model_new.add(Dropout(0.05))
# 24x24x64

model_new.add(SeparableConv2D(128, 3, activation='relu',border_mode='same', use_bias=False)) # RF 11X11
model_new.add(BatchNormalization())
model_new.add(Dropout(0.05))
# 22x22x128

model_new.add(Conv2D(16, (1, 1), use_bias=False)) # RF
model_new.add(MaxPooling2D(pool_size=(2, 2))) 
model_new.add(Dropout(0.05))
# 11x11x16

print("Block2-------------------------------------------------------------------------------------------")

model_new.add(SeparableConv2D(128, 3 ,activation='relu',border_mode='valid', use_bias=False)) 
model_new.add(BatchNormalization())
model_new.add(Dropout(0.05))
#9x9x128

model_new.add(SeparableConv2D(64, 3, activation='relu',border_mode='same', use_bias=False)) 
model_new.add(BatchNormalization())
model_new.add(Dropout(0.05))
#7x7x64

model_new.add(SeparableConv2D(128, 3 ,activation='relu',border_mode='valid', use_bias=False)) 
model_new.add(BatchNormalization())
model_new.add(Dropout(0.05))
#5x5x128


print("Block3-------------------------------------------------------------------------------------------")
model_new.add(SeparableConv2D(64, 3, activation='relu',border_mode='same', use_bias=False))
model_new.add(BatchNormalization())
model_new.add(Dropout(0.05))
# 3x3x64

model_new.add(SeparableConv2D(128, 3, activation='relu',border_mode='valid', use_bias=False)) # RF  
model_new.add(BatchNormalization())
model_new.add(Dropout(0.05))
# 1x1x128

model_new.add(SeparableConv2D(10, 1, 1, use_bias=False)) 
# 1x1x10

print("Block4------------------------------------------------------------------------------------------")


model_new.add(GlobalAveragePooling2D(data_format='channels_last')) # 10
model_new.add(Activation('softmax'))

Model Summary
--
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Block1 (SeparableConv2D)     (None, 32, 32, 64)        219       
_________________________________________________________________
activation_9 (Activation)    (None, 32, 32, 64)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 32, 32, 64)        256       
_________________________________________________________________
dropout_6 (Dropout)          (None, 32, 32, 64)        0         
_________________________________________________________________
block_layer_0 (SeparableConv (None, 30, 30, 64)        4672      
_________________________________________________________________
batch_normalization_2 (Batch (None, 30, 30, 64)        256       
_________________________________________________________________
dropout_7 (Dropout)          (None, 30, 30, 64)        0         
_________________________________________________________________
block_layer_1 (SeparableConv (None, 30, 30, 128)       8768      
_________________________________________________________________
batch_normalization_3 (Batch (None, 30, 30, 128)       512       
_________________________________________________________________
dropout_8 (Dropout)          (None, 30, 30, 128)       0         
_________________________________________________________________
block_layer_2 (SeparableConv (None, 28, 28, 64)        9344      
_________________________________________________________________
batch_normalization_4 (Batch (None, 28, 28, 64)        256       
_________________________________________________________________
dropout_9 (Dropout)          (None, 28, 28, 64)        0         
_________________________________________________________________
block_layer_3 (SeparableConv (None, 28, 28, 128)       8768      
_________________________________________________________________
batch_normalization_5 (Batch (None, 28, 28, 128)       512       
_________________________________________________________________
dropout_10 (Dropout)         (None, 28, 28, 128)       0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 28, 28, 16)        2048      
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 14, 14, 16)        0         
_________________________________________________________________
dropout_11 (Dropout)         (None, 14, 14, 16)        0         
_________________________________________________________________
block_layer_4 (SeparableConv (None, 12, 12, 128)       2192      
_________________________________________________________________
batch_normalization_6 (Batch (None, 12, 12, 128)       512       
_________________________________________________________________
dropout_12 (Dropout)         (None, 12, 12, 128)       0         
_________________________________________________________________
block_layer_5 (SeparableConv (None, 12, 12, 64)        9344      
_________________________________________________________________
batch_normalization_7 (Batch (None, 12, 12, 64)        256       
_________________________________________________________________
dropout_13 (Dropout)         (None, 12, 12, 64)        0         
_________________________________________________________________
block_layer_6 (SeparableConv (None, 10, 10, 128)       8768      
_________________________________________________________________
batch_normalization_8 (Batch (None, 10, 10, 128)       512       
_________________________________________________________________
dropout_14 (Dropout)         (None, 10, 10, 128)       0         
_________________________________________________________________
block_layer_7 (SeparableConv (None, 10, 10, 64)        9344      
_________________________________________________________________
batch_normalization_9 (Batch (None, 10, 10, 64)        256       
_________________________________________________________________
dropout_15 (Dropout)         (None, 10, 10, 64)        0         
_________________________________________________________________
block_layer_8 (SeparableConv (None, 8, 8, 128)         8768      
_________________________________________________________________
batch_normalization_10 (Batc (None, 8, 8, 128)         512       
_________________________________________________________________
dropout_16 (Dropout)         (None, 8, 8, 128)         0         
_________________________________________________________________
block_layer_9 (SeparableConv (None, 8, 8, 10)          1408      
_________________________________________________________________
global_average_pooling2d_1 ( (None, 10)                0         
_________________________________________________________________
activation_10 (Activation)   (None, 10)                0         
=================================================================
Total params: 77,483
Trainable params: 75,563
Non-trainable params: 1,920
_________________________________________________________________




50 EPOCH DATA
--


Epoch 1/50
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:190: The name tf.get_default_session is deprecated. Please use tf.compat.v1.get_default_session instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:197: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:207: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:216: The name tf.is_variable_initialized is deprecated. Please use tf.compat.v1.is_variable_initialized instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:223: The name tf.variables_initializer is deprecated. Please use tf.compat.v1.variables_initializer instead.

390/390 [==============================] - 29s 75ms/step - loss: 1.8902 - acc: 0.2649 - val_loss: 1.4809 - val_acc: 0.4420
Epoch 2/50
390/390 [==============================] - 21s 53ms/step - loss: 1.3984 - acc: 0.4887 - val_loss: 1.1785 - val_acc: 0.5731
Epoch 3/50
390/390 [==============================] - 20s 52ms/step - loss: 1.1479 - acc: 0.5929 - val_loss: 0.9986 - val_acc: 0.6444
Epoch 4/50
390/390 [==============================] - 20s 53ms/step - loss: 1.0093 - acc: 0.6478 - val_loss: 0.8921 - val_acc: 0.6860
Epoch 5/50
390/390 [==============================] - 20s 52ms/step - loss: 0.9000 - acc: 0.6888 - val_loss: 0.8175 - val_acc: 0.7117
Epoch 6/50
390/390 [==============================] - 20s 52ms/step - loss: 0.8177 - acc: 0.7199 - val_loss: 0.7906 - val_acc: 0.7331
Epoch 7/50
390/390 [==============================] - 20s 52ms/step - loss: 0.7585 - acc: 0.7388 - val_loss: 0.6952 - val_acc: 0.7605
Epoch 8/50
390/390 [==============================] - 20s 52ms/step - loss: 0.7094 - acc: 0.7567 - val_loss: 0.6767 - val_acc: 0.7726
Epoch 9/50
390/390 [==============================] - 20s 52ms/step - loss: 0.6694 - acc: 0.7704 - val_loss: 0.6257 - val_acc: 0.7865
Epoch 10/50
390/390 [==============================] - 20s 52ms/step - loss: 0.6334 - acc: 0.7846 - val_loss: 0.6403 - val_acc: 0.7820
Epoch 11/50
390/390 [==============================] - 20s 52ms/step - loss: 0.6098 - acc: 0.7939 - val_loss: 0.6290 - val_acc: 0.7872
Epoch 12/50
390/390 [==============================] - 21s 53ms/step - loss: 0.5927 - acc: 0.7986 - val_loss: 0.6299 - val_acc: 0.7850
Epoch 13/50
390/390 [==============================] - 20s 52ms/step - loss: 0.5624 - acc: 0.8109 - val_loss: 0.6160 - val_acc: 0.7948
Epoch 14/50
390/390 [==============================] - 20s 52ms/step - loss: 0.5458 - acc: 0.8127 - val_loss: 0.5693 - val_acc: 0.8083
Epoch 15/50
390/390 [==============================] - 20s 52ms/step - loss: 0.5293 - acc: 0.8202 - val_loss: 0.5837 - val_acc: 0.8026
Epoch 16/50
390/390 [==============================] - 20s 52ms/step - loss: 0.5141 - acc: 0.8238 - val_loss: 0.5705 - val_acc: 0.8045
Epoch 17/50
390/390 [==============================] - 20s 53ms/step - loss: 0.5033 - acc: 0.8279 - val_loss: 0.6014 - val_acc: 0.8017
Epoch 18/50
390/390 [==============================] - 21s 53ms/step - loss: 0.4880 - acc: 0.8351 - val_loss: 0.6138 - val_acc: 0.7935
Epoch 19/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4756 - acc: 0.8395 - val_loss: 0.5631 - val_acc: 0.8140
Epoch 20/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4659 - acc: 0.8406 - val_loss: 0.5771 - val_acc: 0.8110
Epoch 21/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4591 - acc: 0.8464 - val_loss: 0.5849 - val_acc: 0.8112
Epoch 22/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4504 - acc: 0.8468 - val_loss: 0.5786 - val_acc: 0.8082
Epoch 23/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4351 - acc: 0.8525 - val_loss: 0.5796 - val_acc: 0.8111
Epoch 24/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4318 - acc: 0.8519 - val_loss: 0.5826 - val_acc: 0.8143
Epoch 25/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4199 - acc: 0.8567 - val_loss: 0.5697 - val_acc: 0.8154
Epoch 26/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4096 - acc: 0.8599 - val_loss: 0.6035 - val_acc: 0.8070
Epoch 27/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4083 - acc: 0.8612 - val_loss: 0.5714 - val_acc: 0.8201
Epoch 28/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4090 - acc: 0.8621 - val_loss: 0.5558 - val_acc: 0.8224
Epoch 29/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3915 - acc: 0.8656 - val_loss: 0.5608 - val_acc: 0.8248
Epoch 30/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3889 - acc: 0.8689 - val_loss: 0.5729 - val_acc: 0.8197
Epoch 31/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3807 - acc: 0.8698 - val_loss: 0.5699 - val_acc: 0.8219
Epoch 32/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3800 - acc: 0.8709 - val_loss: 0.5813 - val_acc: 0.8150
Epoch 33/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3771 - acc: 0.8737 - val_loss: 0.5818 - val_acc: 0.8169
Epoch 34/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3757 - acc: 0.8731 - val_loss: 0.5534 - val_acc: 0.8265
Epoch 35/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3663 - acc: 0.8753 - val_loss: 0.5391 - val_acc: 0.8282
Epoch 36/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3550 - acc: 0.8791 - val_loss: 0.5897 - val_acc: 0.8190
Epoch 37/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3533 - acc: 0.8807 - val_loss: 0.5862 - val_acc: 0.8210
Epoch 38/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3551 - acc: 0.8808 - val_loss: 0.5631 - val_acc: 0.8206
Epoch 39/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3459 - acc: 0.8845 - val_loss: 0.5804 - val_acc: 0.8242
Epoch 40/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3451 - acc: 0.8840 - val_loss: 0.5671 - val_acc: 0.8231
Epoch 41/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3444 - acc: 0.8843 - val_loss: 0.5836 - val_acc: 0.8201
Epoch 42/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3349 - acc: 0.8855 - val_loss: 0.5821 - val_acc: 0.8275
Epoch 43/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3349 - acc: 0.8890 - val_loss: 0.5659 - val_acc: 0.8279
Epoch 44/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3262 - acc: 0.8916 - val_loss: 0.6030 - val_acc: 0.8227
Epoch 45/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3342 - acc: 0.8884 - val_loss: 0.5728 - val_acc: 0.8271
Epoch 46/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3308 - acc: 0.8894 - val_loss: 0.5907 - val_acc: 0.8275
Epoch 47/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3197 - acc: 0.8934 - val_loss: 0.5652 - val_acc: 0.8272
Epoch 48/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3207 - acc: 0.8931 - val_loss: 0.5616 - val_acc: 0.8266
Epoch 49/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3115 - acc: 0.8963 - val_loss: 0.5841 - val_acc: 0.8216
Epoch 50/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3137 - acc: 0.8951 - val_loss: 0.6351 - val_acc: 0.8248
Model took 1024.95 seconds to train
