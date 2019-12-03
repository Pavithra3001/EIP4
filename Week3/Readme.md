


Given Model Accuracy - Accuracy on test data is: 83.03
Derived Model Accuracy - Accuracy on test data is: 83.15


Model Definition
--

from keras.layers.convolutional import DepthwiseConv2D, SeparableConv2D, Conv2D
from keras.layers import Activation, GlobalAveragePooling2D, AveragePooling2D

   

model_new = Sequential()

model_new.add(SeparableConv2D(64, 3, input_shape=(32, 32, 3),border_mode='same', use_bias=False, name = "Block1")) # RF 1x1
model_new.add(Activation('relu'))
model_new.add(BatchNormalization())
model_new.add(Dropout(0.05))
# 32x32x64 


model_new.add(SeparableConv2D(64, 3 ,activation='relu',border_mode='valid', use_bias=False)) # RF 3x3
model_new.add(BatchNormalization())
model_new.add(Dropout(0.05))
# 30x30x64

model_new.add(SeparableConv2D(128, 3, activation='relu',border_mode='same', use_bias=False)) # RF 5x5
model_new.add(BatchNormalization())
model_new.add(Dropout(0.05))
# 30x30x128


model_new.add(SeparableConv2D(64, 3 ,activation='relu',border_mode='valid', use_bias=False)) # RF 7x7
model_new.add(BatchNormalization())
model_new.add(Dropout(0.05))
# 28x28x64

model_new.add(SeparableConv2D(128, 3, activation='relu',border_mode='same', use_bias=False)) # RF 9x9
model_new.add(BatchNormalization())
model_new.add(Dropout(0.05))
# 28x28x128

model_new.add(Conv2D(16, (1, 1), use_bias=False)) # RF 11x11
# 28x28x16
model_new.add(MaxPooling2D(pool_size=(2, 2))) # RF 12x12
model_new.add(Dropout(0.05))
# 14x14x16


model_new.add(SeparableConv2D(128, 3 ,activation='relu',border_mode='valid', use_bias=False))  # RF 16x16
model_new.add(BatchNormalization())
model_new.add(Dropout(0.05))
# 12x12x128

model_new.add(SeparableConv2D(64, 3, activation='relu',border_mode='same', use_bias=False))  # RF 20x20
model_new.add(BatchNormalization())
model_new.add(Dropout(0.05))
# 12x12x64

model_new.add(SeparableConv2D(128, 3 ,activation='relu',border_mode='valid', use_bias=False)) # RF 24x24
model_new.add(BatchNormalization())
model_new.add(Dropout(0.05))
# 10x10x128


print("Block3-------------------------------------------------------------------------------------------")
model_new.add(SeparableConv2D(64, 3, activation='relu',border_mode='same', use_bias=False)) # RF 28x28
model_new.add(BatchNormalization())
model_new.add(Dropout(0.05))
# 10x10x64

model_new.add(SeparableConv2D(128, 3, activation='relu',border_mode='valid', use_bias=False)) # RF 32x32
model_new.add(BatchNormalization())
model_new.add(Dropout(0.05))
# 8x8x128

model_new.add(SeparableConv2D(10, 1, 1, use_bias=False)) 
# 8x8x10

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

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
390/390 [==============================] - 82s 211ms/step - loss: 1.3360 - acc: 0.5109 - val_loss: 4.9353 - val_acc: 0.2891
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
390/390 [==============================] - 78s 200ms/step - loss: 0.9027 - acc: 0.6782 - val_loss: 1.1121 - val_acc: 0.6375
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
390/390 [==============================] - 77s 197ms/step - loss: 0.7510 - acc: 0.7347 - val_loss: 0.7956 - val_acc: 0.7313
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
390/390 [==============================] - 78s 200ms/step - loss: 0.6634 - acc: 0.7682 - val_loss: 0.7492 - val_acc: 0.7412
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
390/390 [==============================] - 78s 201ms/step - loss: 0.6037 - acc: 0.7890 - val_loss: 0.6385 - val_acc: 0.7824
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
390/390 [==============================] - 78s 201ms/step - loss: 0.5574 - acc: 0.8043 - val_loss: 0.6697 - val_acc: 0.7742
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
390/390 [==============================] - 78s 201ms/step - loss: 0.5203 - acc: 0.8190 - val_loss: 0.6349 - val_acc: 0.7891
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
390/390 [==============================] - 78s 201ms/step - loss: 0.4934 - acc: 0.8260 - val_loss: 0.5980 - val_acc: 0.8013
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
390/390 [==============================] - 79s 202ms/step - loss: 0.4629 - acc: 0.8383 - val_loss: 0.6048 - val_acc: 0.8027
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
390/390 [==============================] - 79s 201ms/step - loss: 0.4408 - acc: 0.8451 - val_loss: 0.5820 - val_acc: 0.8092
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
390/390 [==============================] - 79s 201ms/step - loss: 0.4229 - acc: 0.8498 - val_loss: 0.5585 - val_acc: 0.8166
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
390/390 [==============================] - 78s 201ms/step - loss: 0.4056 - acc: 0.8582 - val_loss: 0.5804 - val_acc: 0.8146
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
390/390 [==============================] - 78s 200ms/step - loss: 0.3867 - acc: 0.8640 - val_loss: 0.5684 - val_acc: 0.8187
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
390/390 [==============================] - 78s 201ms/step - loss: 0.3771 - acc: 0.8667 - val_loss: 0.5658 - val_acc: 0.8167
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
390/390 [==============================] - 78s 200ms/step - loss: 0.3620 - acc: 0.8723 - val_loss: 0.5767 - val_acc: 0.8174
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
390/390 [==============================] - 78s 200ms/step - loss: 0.3523 - acc: 0.8750 - val_loss: 0.5808 - val_acc: 0.8151
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
390/390 [==============================] - 78s 200ms/step - loss: 0.3379 - acc: 0.8805 - val_loss: 0.5631 - val_acc: 0.8203
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
390/390 [==============================] - 78s 201ms/step - loss: 0.3283 - acc: 0.8844 - val_loss: 0.5703 - val_acc: 0.8219
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
390/390 [==============================] - 78s 200ms/step - loss: 0.3198 - acc: 0.8859 - val_loss: 0.5556 - val_acc: 0.8271
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
390/390 [==============================] - 78s 200ms/step - loss: 0.3117 - acc: 0.8889 - val_loss: 0.5750 - val_acc: 0.8280
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0004065041.
390/390 [==============================] - 78s 200ms/step - loss: 0.3008 - acc: 0.8940 - val_loss: 0.5753 - val_acc: 0.8226
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.000389661.
390/390 [==============================] - 78s 200ms/step - loss: 0.3007 - acc: 0.8918 - val_loss: 0.5647 - val_acc: 0.8284
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0003741581.
390/390 [==============================] - 78s 200ms/step - loss: 0.2917 - acc: 0.8960 - val_loss: 0.5708 - val_acc: 0.8258
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0003598417.
390/390 [==============================] - 78s 200ms/step - loss: 0.2833 - acc: 0.8999 - val_loss: 0.5745 - val_acc: 0.8296
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.0003465804.
390/390 [==============================] - 78s 200ms/step - loss: 0.2780 - acc: 0.9030 - val_loss: 0.5747 - val_acc: 0.8287
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0003342618.
390/390 [==============================] - 78s 200ms/step - loss: 0.2724 - acc: 0.9012 - val_loss: 0.5700 - val_acc: 0.8312
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.0003227889.
390/390 [==============================] - 78s 200ms/step - loss: 0.2663 - acc: 0.9047 - val_loss: 0.5853 - val_acc: 0.8293
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.0003120774.
390/390 [==============================] - 78s 200ms/step - loss: 0.2646 - acc: 0.9055 - val_loss: 0.5873 - val_acc: 0.8294
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.000302054.
390/390 [==============================] - 78s 200ms/step - loss: 0.2529 - acc: 0.9088 - val_loss: 0.5897 - val_acc: 0.8263
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0002926544.
390/390 [==============================] - 76s 195ms/step - loss: 0.2545 - acc: 0.9081 - val_loss: 0.5994 - val_acc: 0.8251
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0002838221.
390/390 [==============================] - 76s 195ms/step - loss: 0.2491 - acc: 0.9102 - val_loss: 0.5953 - val_acc: 0.8289
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.0002755074.
390/390 [==============================] - 76s 195ms/step - loss: 0.2415 - acc: 0.9136 - val_loss: 0.5940 - val_acc: 0.8324
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.000267666.
390/390 [==============================] - 76s 195ms/step - loss: 0.2384 - acc: 0.9151 - val_loss: 0.6063 - val_acc: 0.8279
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0002602585.
390/390 [==============================] - 76s 195ms/step - loss: 0.2369 - acc: 0.9153 - val_loss: 0.5921 - val_acc: 0.8296
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.00025325.
390/390 [==============================] - 76s 195ms/step - loss: 0.2330 - acc: 0.9150 - val_loss: 0.6060 - val_acc: 0.8285
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0002466091.
390/390 [==============================] - 76s 195ms/step - loss: 0.2298 - acc: 0.9157 - val_loss: 0.6026 - val_acc: 0.8297
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0002403076.
390/390 [==============================] - 76s 195ms/step - loss: 0.2309 - acc: 0.9178 - val_loss: 0.5971 - val_acc: 0.8336
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0002343201.
390/390 [==============================] - 76s 196ms/step - loss: 0.2229 - acc: 0.9196 - val_loss: 0.6109 - val_acc: 0.8320
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0002286237.
390/390 [==============================] - 76s 196ms/step - loss: 0.2233 - acc: 0.9202 - val_loss: 0.6078 - val_acc: 0.8289
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0002231977.
390/390 [==============================] - 76s 196ms/step - loss: 0.2179 - acc: 0.9219 - val_loss: 0.6214 - val_acc: 0.8303
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0002180233.
390/390 [==============================] - 76s 196ms/step - loss: 0.2158 - acc: 0.9222 - val_loss: 0.6325 - val_acc: 0.8268
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0002130833.
390/390 [==============================] - 76s 196ms/step - loss: 0.2160 - acc: 0.9214 - val_loss: 0.6252 - val_acc: 0.8290
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0002083623.
390/390 [==============================] - 76s 196ms/step - loss: 0.2089 - acc: 0.9255 - val_loss: 0.6211 - val_acc: 0.8333
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0002038459.
390/390 [==============================] - 76s 196ms/step - loss: 0.2086 - acc: 0.9248 - val_loss: 0.6320 - val_acc: 0.8274
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0001995211.
390/390 [==============================] - 76s 196ms/step - loss: 0.2046 - acc: 0.9253 - val_loss: 0.6311 - val_acc: 0.8304
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0001953761.
390/390 [==============================] - 76s 196ms/step - loss: 0.2050 - acc: 0.9255 - val_loss: 0.6349 - val_acc: 0.8291
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0001913998.
390/390 [==============================] - 76s 196ms/step - loss: 0.2000 - acc: 0.9267 - val_loss: 0.6341 - val_acc: 0.8319
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0001875821.
390/390 [==============================] - 76s 196ms/step - loss: 0.2015 - acc: 0.9280 - val_loss: 0.6288 - val_acc: 0.8335
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0001839137.
390/390 [==============================] - 76s 196ms/step - loss: 0.2024 - acc: 0.9264 - val_loss: 0.6476 - val_acc: 0.8296
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.000180386.
390/390 [==============================] - 76s 195ms/step - loss: 0.1994 - acc: 0.9265 - val_loss: 0.6363 - val_acc: 0.8315
Model took 3873.17 seconds to train

Accuracy on test data is: 83.15
