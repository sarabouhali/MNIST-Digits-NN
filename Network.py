from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import to_categorical
from scipy.io import loadmat, savemat
import json


class NN:

    def load_image(self, dataset, i):  # Function to load a single image
        (x_train, y_train), (x_test, y_test) = dataset.load_data()
        return x_test[i]

    def prepare_data(self, dataset, x_size, num_classes): # Function to prepare the training and testing data

        (x_train, y_train), (x_test, y_test) = dataset.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        # Flatten the images
        x_train = x_train.reshape(x_train.shape[0], x_size)
        x_test = x_test.reshape(x_test.shape[0], x_size)

        # Convert to "one-hot" vectors using the to_categorical function
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        return x_train, x_test, y_train, y_test

    def model_architecture(self, x_size, num_classes, hidden_units, hidden_activation, W_init):  # Function to define our neural network's architecture
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=hidden_units, activation=hidden_activation, kernel_initializer=W_init, input_shape=(x_size,)),
            tf.keras.layers.Dense(units=num_classes, kernel_initializer=W_init, activation='softmax')
        ])
        print(model.summary())
        return model

    def save_model(self, model, json_path, weight_path):  # Function to save the model and its weights
        json_string = model.to_json()
        open(json_path, 'w').write(json_string)
        dict = {}
        i = 0
        for layer in model.layers:
            weights = layer.get_weights()
            my_list = np.zeros(len(weights), dtype=np.object)
            my_list[:] = weights
            dict[str(i)] = my_list
            i += 1

        savemat(weight_path, dict)

    def load_model(self, json_path):  # Function to load the model
        model = tf.keras.models.model_from_json(open(json_path).read())
        return model

    def load_weights(self, model, weight_path):  # Function to load the model weights
        def conv_dict(dict2):
            i = 0
            dict = {}
            for i in range(len(dict2)):
                if str(i) in dict2:
                    if dict2[str(i)].shape == (0, 0):
                        dict[str(i)] = dict2[str(i)]
                    else:
                        weights = dict2[str(i)][0]
                        weights2 = []
                        for weight in weights:
                            if weight.shape in [(1, x) for x in range(0, 5000)]:
                                weights2.append(weight[0])
                            else:
                                weights2.append(weight)
                        dict[str(i)] = weights2
            return dict

        dict2 = loadmat(weight_path)
        dict = conv_dict(dict2)
        i = 0
        for layer in model.layers:
            weights = dict[str(i)]
            layer.set_weights(weights)
            i += 1
        return model

    def save_hidden_unit(self, model, json_path, iteration):  # Function to save the hidden unit activation and weight
        dict = {}
        weights = model.layers[1].get_weights()[1]
        my_list = np.zeros(len(weights), dtype=np.object)
        my_list[:] = weights
        dict[iteration] = my_list

        with open(json_path, 'a') as outfile:
            json.dump(dict, outfile)

    def split_val(self, x_train, y_train, validation_split): # Function to precise a proportion of data for validation
        l = len(x_train)

        if 0. < validation_split < 1.:
            val_samples = int(l * validation_split)
            x_val = x_train[l - val_samples:]
            y_val = y_train[l - val_samples:]

            x_train = x_train[:l-val_samples]
            y_train = y_train[:l - val_samples]

        else :
            print ("Validation split must be between 0 and 1!")
        return x_train, y_train, x_val, y_val

    def train(self, model, trainx, trainy, val_split, batchsize): # Function to train our model iteratively
        num_iters = int((len(trainy)+len(trainx))/batchsize)
        total_iterations = 0
        time_before = datetime.now()
        w = []
        loss = []
        val_loss = []
        accuracy = []
        val_acc = []
        print("Starting training ...")

        def load_train_batch(batch_size, train_x, train_y):
            print("Loading batch training data ...")
            Num_samples = len(train_x)
            list_iter = np.random.permutation(Num_samples)
            list_iter = list_iter[Num_samples - batch_size:]
            inputs = []
            targets = []
            c = -1
            for i in list_iter:
                c += 1
                if c == 0:
                    inputs = train_x[i]
                    targets = train_y[i]
                elif c > 0:
                    inputs = np.vstack((inputs, train_x[i]))
                    targets = np.vstack((targets, train_y[i]))

            print("========== Training data loaded ==========")
            return inputs, targets

        for it_num in range(num_iters):
            inputs, targets = load_train_batch(batchsize, trainx, trainy)
            T_inputs, T_targets, V_inputs, V_targets = self.split_val(inputs, targets, val_split)
            batch_loss = model.train_on_batch(T_inputs, T_targets)
            v_batch_loss = model.test_on_batch(V_inputs, V_targets)

            total_iterations += 1
            #if it_num % 200 == 0 and it_num != 0:
            print("Iteration ==> " + str(total_iterations) + " took: " + str(datetime.now() - time_before) + ", with loss of " + str(batch_loss[0]) + " and accuracy of " + str(batch_loss[1]))
            print("                    validation set loss of "+ str(v_batch_loss[0]) + " and validation accuracy of " + str(v_batch_loss[1]))
            weights = model.layers[1].get_weights()[0]
            my_list = np.zeros((32, 10), dtype=np.object)
            my_list[:, :] = weights
            w.append(my_list)
            loss.append(batch_loss[0])
            val_loss.append(v_batch_loss[0])
            accuracy.append(batch_loss[1])
            val_acc.append(v_batch_loss[1])

        print("========== Training finished ==========")
        print()

        self.save_model(model, "output/model.json", "output/weights.mat")

        return w, loss, val_loss, accuracy, val_acc

    def getActivations(self, x_test, model):  # Function to save the activation function values for the hidden layer

        x = 28*28

        model2 = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=32, input_shape=(x,), weights=model.layers[0].get_weights() ,activation='relu')
        ])

        activations = model2.predict(x_test)
        return activations







