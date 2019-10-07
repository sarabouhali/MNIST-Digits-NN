import Network as n
import plots

mnist = n.tf.keras.datasets.mnist

image_vector_size = 28 * 28
num_classes = 10

network = n.NN()
x_train, x_test, y_train, y_test = network.prepare_data(mnist, image_vector_size, num_classes)

model = network.load_model("output/model.json")

network.load_weights(model, "output/weights.mat")

imageToUse = network.load_image(mnist, 5)
'''plt.imshow(n.np.reshape(imageToUse,[28,28]), interpolation="nearest", cmap="gray")
plt.show()'''

network.getActivations(imageToUse, model)

plot = plots.DataViz()



