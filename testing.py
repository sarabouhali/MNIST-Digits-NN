import Network as n
import plots

mnist = n.tf.keras.datasets.mnist

image_vector_size = 28 * 28
num_classes = 10

network = n.NN()
x_train, x_test, y_train, y_test = network.prepare_data(mnist, image_vector_size, num_classes)

model = network.load_model("output/model.json")

network.load_weights(model, "output/weights.mat")

x = network.getActivations(x_train, model)
y = network.getActivations2(x_train, model)

plot = plots.DataViz()

plot.plotHist(x, 3, 1)
#plot.plotHist(y, 1, 2)
plot.plotHist(x, 2, 1)
#plot.plotHist(y, 2, 2)
f = network.predict_class(8, model, mnist)
print("the class is", f)
