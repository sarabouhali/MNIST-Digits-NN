import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

mnist = tf.keras.datasets.mnist

# tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
# print_weights = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: print(model.layers[0].get_weights()))

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten the images
image_vector_size = 28 * 28
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)

# Convert to "one-hot" vectors using the to_categorical function
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print(type(x_train[:2]))
print(y_train[:2])



'''
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=32, activation='relu', kernel_initializer='glorot_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=(image_vector_size,)),
    # tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(units=num_classes, kernel_initializer='glorot_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='softmax')
])

model.summary()

sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print("Starting training ...")

#checkpoint = ModelCheckpoint("weights.hdf5", monitor='loss', verbose=1, save_weights_only=True, save_best_only=False, mode='auto', period=1)

history = model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=1, validation_split=.1)

loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

print('Test loss: ', loss)
print('Test accuracy: ', accuracy)

print(model.layers[1].get_weights()[1])
'''
