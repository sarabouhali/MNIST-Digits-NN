import Network as n
import plots

mnist = n.tf.keras.datasets.mnist

image_vector_size = 28 * 28
num_classes = 10

network = n.NN()
x_train, x_test, y_train, y_test = network.prepare_data(mnist, image_vector_size, num_classes)

model = network.model_architecture(image_vector_size, num_classes, hidden_units=32, hidden_activation='relu',
                                   W_init='glorot_normal')

sgd = n.tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

w, t_loss, v_loss, acc, val_acc = network.train(trainx=x_train, trainy=y_train, model=model, val_split=0.3,
                                                     batchsize=128)

print("Evaluating the model")
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print('Test loss: ', loss)
print('Test accuracy: ', accuracy)

plot = plots.DataViz()

plot.plotLoss(t_loss, v_loss)
plot.plotWeights(w, 0)
plot.plotAcc(acc, val_acc)

