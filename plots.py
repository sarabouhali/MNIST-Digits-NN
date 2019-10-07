import matplotlib.pyplot as plt
import numpy as np


class DataViz:

    def plotHist(self):

        return

    def plotLoss(self, loss, val_loss):

        plt.plot(loss)
        plt.plot(val_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('iteration')
        plt.legend(['training', 'validation'], loc='best')
        plt.savefig("output/figs/Lossperiteration2.png")
        plt.show()
        return

    def plotAcc(self, history):
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='best')
        plt.savefig("output/figs/Accperepoch.png")
        plt.show()

    def plotWeights(self, w):
        def unit_vector(vector):
            """ Returns the unit vector of the vector.  """
            return vector / np.linalg.norm(vector)

        def angle_between(v1, v2):
            """ Returns the angle in radians between vectors 'v1' and 'v2'::
            """
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        def angles_w(weights):
            ang=[]
            for i in range(len(weights)-1):
                ang.append(angle_between(weights[i], weights[i+1]))
            return ang

        a = angles_w(w)
        x = [i for i in range(1, len(a)+1)]
        plt.plot(x, a)
        plt.title('weight changes for a hidden unit')
        plt.ylabel('angle')
        plt.xlabel('iteration')
        plt.savefig("output/figs/weights2.png")
        plt.show()

        return