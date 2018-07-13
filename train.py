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

nn = NN(layers = [784, 512, 10], activations = ['relu', 'sigmoid'])
nn.train(x[0:2], y[0:2], step_size = 0.0001, epochs = 10000, batch_size = 2, method = "gradient_descent", notification_frequency = 100)
nn.save()
