# MNIST-Digits-NN

## Dataset description :

We chose to work with the MNIST dataset which is an open source dataset that contains 70,000 images depicting handwritten digits from 250 different people, 50 percent high school students, and 50 percent employees from the Census Bureau. 

The images of this dataset have a size of 28 x 28 pixel where each pixel has its own grayscale level ranging from 0 to 255. 
The number of classes is 10 referring to the digits we want our model to recognize {0,1,2,3,4,5,6,7,8,9}.
The figure below represents a random selection of MNIST digits on which the training and testing of the model will be done.

<p align="center">
  <img width="460" height="300" src="https://github.com/Sarabouh/MNIST-Digits-NN/blob/master/MNIST.png">
</p>


## Data preprocessing :

Data preprocessing is an important phase before modeling the network. We will convert the input images into a suitable representation that can be fed to our model. 
In order to unify and compress the properties of the images in the dataset, we scale them by dividing their intensities by 255 since we are working w gray scale images and that way the intensity values will range from 0 to 1. 
We then flatten the images into a vector representation where each image will be converted into a 784 dimensional vector. This flattening process is not ideal for the reason that it obfuscate information about the pixels locations but it won’t affect the model’s performance much. 

We converted the target classes data into “one-hot” encoded vectors since the labels for this dataset are numerical values from 0 to 9 and it’s important that our algorithm treats these as items in a set, rather than ordinal values. In our dataset the value “0” isn’t smaller than the value “9”, they are just two different class labels. The one-hot version of 5 is the following list [0, 0, 0, 0, 0, 1, 0, 0, 0, 0].


## Network Setup :

We experimented with a 3-layer fully connected neural network, the first layer being the input layer has 784 input nodes so as to receive the output of the data preprocessing step which results in a 784-dimension vector per image. 
The second layer is a hidden layer comprising 450 units. It serves as an initial bottleneck to reduce the dimensionality of the input information without losing its meaningfulness. Moreover, this layer uses  the Rectifier Linear Unit ReLU as an activation function.
The Third layer is a hidden layer comprising 32 units. It uses  the Sigmoid as an activation function.
The output layer has 10 units that will provide a one hot encoded vector for each image being fed to the model; it uses the Softmax activation function. 
In the implementation of our neural network, we used the following parameter settings for training the model over the MNIST dataset :
◦	The learning rate: 0.01, the momentum: 0.9
◦	The number of iterations: 2808
◦	The batch size : 128
◦	Number of hidden layers : 2
◦	The number of nodes : 
Input layer : 784 nodes (28*28 pixels)
Hidden layer : 450 nodes
Hidden layer : 32 nodes
Output layer : 10 nodes (10 classes)
For this setup, we used the existing toolbox named “Keras” and to build better visualisations we used the library “Plotly”.
