from NeuralNetwork import NeuralNetwork as NN
import numpy as np
import mnist

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

test_imgs = mnist.test_images()
test_lbls = mnist.test_labels()

x = test_imgs.reshape((test_imgs.shape[0], test_imgs.shape[1] * test_imgs.shape[2]))
y = np.zeros((test_lbls.shape[0], 10))
for i in range(y.shape[0]):
    y[i][test_lbls[i]] = 1

x = x / 255

nn = NN(layers = [784, 800, 10], activations = ['sigmoid', 'sigmoid'])
nn.load()

res = vout(nn.fprop(x))

print("Total error: ")
print(np.sum(nn.error(y)))
print("Number correct: ")
print(num_correct(res, y))
print(corr(res, y)[0:20])
