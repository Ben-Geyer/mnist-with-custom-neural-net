from NeuralNetwork import NeuralNetwork as NN
import cv2
import numpy as np
import os

def convert_output(o):
    if (o >= .5):
        return 1
    else:
        return 0

def num_correct(ans, res):
    num_corr = 0
    for i in range(ans.shape[0]):
        if (np.array_equal(ans[i], res[i])):
            num_corr += 1
    return num_corr

def corr(ans, res):
    num_corr = np.zeros(ans.shape[0])
    for i in range(ans.shape[0]):
        if (np.array_equal(ans[i], res[i])):
            num_corr[i] = 1
    return num_corr

vout = np.vectorize(convert_output)

x = np.zeros((10, 784))

for i in range(10):
    img = cv2.imread(os.path.join('custom_imgs', str(i + 1) + '.resized.JPG'), 0)
    img = cv2.bitwise_not(img)

    if(img.shape[0] < 28):
        img = np.append(img, np.zeros(((28 - img.shape[0]), 28)), axis = 0)
    elif(img.shape[1] < 28):
        img = np.append(img, np.zeros((28, (28 - img.shape[1]))), axis = 1)

    x[i] += np.reshape(img, (784,)) / 255

nn = NN(layers = [784, 800, 10], activations = ['sigmoid', 'softmax'])
nn.load()

res = vout(nn.fprop(x))

print("Predictions: ")
print(res)
