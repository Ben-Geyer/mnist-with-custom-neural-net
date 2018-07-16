from NeuralNetwork import NeuralNetwork as NN
import numpy as np
import mnist

train_imgs = mnist.train_images()
train_lbls = mnist.train_labels()

x = train_imgs.reshape((train_imgs.shape[0], train_imgs.shape[1] * train_imgs.shape[2]))
y = np.zeros((train_lbls.shape[0], 10))
for i in range(y.shape[0]):
    y[i][train_lbls[i]] = 1

x = x / 255

nn = NN(layers = [784, 800, 10], activations = ['sigmoid', 'softmax'])
nn.train(x, y, step_size = 0.001, epochs = 56250, batch_size = 16, method = "adam", notification_frequency = 100)
nn.save()
