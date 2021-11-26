from tensorflow import keras
from sklearn.model_selection import train_test_split

(xtrain, xtarget), (ytrain, ytarget) = keras.datasets.fashion_mnist.load_data()

xtrain = xtrain.reshape(-1,28,28,1)/255.0
ytrain = ytrain.reshape(-1,28,28,1)
train, val, target, val_target = train_test_split(xtrain, xtarget)


model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation='softmax'))

print(train.shape, target.shape)
print(val.shape, val_target.shape)
model.summary()

'''
(45000, 28, 28, 1) (45000,)
(15000, 28, 28, 1) (15000,)
Model: "sequential_12"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_24 (Conv2D)           (None, 28, 28, 32)        320       
_________________________________________________________________
max_pooling2d_24 (MaxPooling (None, 14, 14, 32)        0         
_________________________________________________________________
conv2d_25 (Conv2D)           (None, 14, 14, 64)        18496     
_________________________________________________________________
max_pooling2d_25 (MaxPooling (None, 7, 7, 64)          0         
_________________________________________________________________
flatten_11 (Flatten)         (None, 3136)              0         
_________________________________________________________________
dense_21 (Dense)             (None, 100)               313700    
_________________________________________________________________
dropout_11 (Dropout)         (None, 100)               0         
_________________________________________________________________
dense_22 (Dense)             (None, 10)                1010      
=================================================================
Total params: 333,526
Trainable params: 333,526
Non-trainable params: 0
_________________________________________________________________
'''

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best_checkpoint.h5')
earlystop_cb = keras.callbacks.EarlyStopping(patience=2)
hist = model.fit(xtrain, xtarget, epochs=10, callbacks=[checkpoint_cb, earlystop_cb], validation_data=(val, val_target))

'''
Epoch 1/10
1407/1407 [==============================] - 11s 7ms/step - loss: 0.1700 - accuracy: 0.9360 - val_loss: 0.2471 - val_accuracy: 0.9231
Epoch 2/10
1407/1407 [==============================] - 11s 8ms/step - loss: 0.1573 - accuracy: 0.9398 - val_loss: 0.2491 - val_accuracy: 0.9255
Epoch 3/10
1407/1407 [==============================] - 11s 8ms/step - loss: 0.1485 - accuracy: 0.9431 - val_loss: 0.2363 - val_accuracy: 0.9258
Epoch 4/10
1407/1407 [==============================] - 10s 7ms/step - loss: 0.1447 - accuracy: 0.9457 - val_loss: 0.2573 - val_accuracy: 0.9211
Epoch 5/10
1407/1407 [==============================] - 11s 8ms/step - loss: 0.1360 - accuracy: 0.9476 - val_loss: 0.2919 - val_accuracy: 0.9157
'''

realmodel = keras.models.load_model('best_checkpoint.h5')
realmodel.evaluate(train, target)
realmodel.evaluate(test, ttt)
realmodel.evaluate(val, val_target)

'''
1407/1407 [==============================] - 5s 4ms/step - loss: 0.1094 - accuracy: 0.9556
313/313 [==============================] - 1s 4ms/step - loss: 79.7146 - accuracy: 0.8739
469/469 [==============================] - 2s 4ms/step - loss: 0.2919 - accuracy: 0.9157
[0.2918599545955658, 0.9157333374023438]
'''
n = 5
result = realmodel.predict(test[:n])
print(np.argmax(result, axis=1))
print(ytarget[:n])

false_list = np.argmax(result, axis=1) == ttt[:n]
print(false_list)

'''
[9 2 1 2 6]
[9 2 1 1 6]
[ True  True  True  False  True]
'''

