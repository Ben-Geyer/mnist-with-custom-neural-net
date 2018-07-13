import numpy as np
import pickle as pk


#TODO: Add softmax
#TODO: Add cross enropy loss


class NeuralNetwork(object):
    '''General class for creating a Neural Network with numpy'''



    def __init__(self, layers, activations):
        #Make sure there are enough layers to make a valid network
        if(len(layers) > 1):
            self.layers = layers
        else:
            print("Enter a valid number of layers -- greater than or equal to two. Terminating Neural Network.")
            self.terminate()

        #Set layers
        self.nlayers = len(self.layers)
        self.l = [0 for i in range(self.nlayers)]

        #Check activations
        if(len(activations) == self.nlayers - 1):
            self.activations = activations
        else:
            print("Enter exactly one less activation than layers in your network. Terminating Neural Network.")
            self.terminate()

        self.randomize(True)


    def randomize(self, zeros = False, seed = 1):
        #Store seed
        np.random.seed(seed)

        #Set size of list storing synapses (weights)
        self.syn = [0 for i in range(self.nlayers - 1)]

        #Set size of list for bias neurons
        self.bias = [0 for i in range(self.nlayers - 1)]

        #Set all weights and bias neurons
        for i in range(self.nlayers - 1):
            if(zeros == True):
                self.syn[i] = np.zeros((self.layers[i], self.layers[i + 1]))
                self.bias[i] = np.zeros((1, self.layers[i + 1]))
            else:
                self.syn[i] = 2 * np.random.rand(self.layers[i], self.layers[i + 1]) - 1
                self.bias[i] = 2 * np.random.rand(1, self.layers[i + 1]) - 1


    def sigmoid(self, x, deriv = False):
        if(deriv == True):
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))


    def tanh(self, x, deriv = False):
        if(deriv == True):
            return 1 - (x ** 2)
        return np.tanh(x)


    def relu(self, x, deriv = False):
        def f(x):
            if(x < 0):
                return 0
            return x

        def df(x):
            if(x >= 0):
                return 1
            return 0

        vf = np.vectorize(f)
        vdf = np.vectorize(df)

        if(deriv == True):
            return vdf(x)
        return vf(x)


    def error(self, y, deriv = False):
        if(deriv == True):
            return 2 * (self.l[-1] - y)
        return (self.l[-1] - y) ** 2


    def fprop(self, x):
        self.l[0] = x

        for i in range(self.nlayers - 1):
            self.l[i + 1] = getattr(self, self.activations[i])(np.dot(self.l[i], self.syn[i]) + self.bias[i])

        return self.l[-1]


    def grad(self, y):
        l_delta = [0 for i in range(self.nlayers - 1)]
        w_delta = [0 for i in range(self.nlayers - 1)]
        b_delta = [0 for i in range(self.nlayers - 1)]

        l_delta[-1] = self.error(y, True) * getattr(self, self.activations[-1])(self.l[-1], True)

        for i in range(self.nlayers - 2):
            l_delta[-2 - i] = np.dot(l_delta[-1 - i], self.syn[-1 - i].T) * getattr(self, self.activations[-2 - i])(self.l[-2 - i], True)

        for i in range(self.nlayers - 1):
            w_delta[i] = np.dot(self.l[i].T, l_delta[i]) * (1 / y.shape[0])
            b_delta[i] = np.sum(l_delta[i], axis = 0) * (1 / y.shape[0])

        return w_delta, b_delta


    def gradient_descent(self, y, step_size = 0.01):
        w_delta, b_delta = self.grad(y)

        for i in range(self.nlayers - 1):
            self.syn[i] -= step_size * w_delta[i]
            self.bias[i] -= step_size * b_delta[i]


    def momentum(self, y, step_size = 1, momentum = 0.9):
        if getattr(self, "vel_weights", None) is None:
            self.vel_weights = [0 for i in range(self.nlayers - 1)]
        if getattr(self, "vel_bias", None) is None:
            self.vel_bias = [0 for i in range(self.nlayers - 1)]

        w_delta, b_delta = self.grad(y)

        for i in range(self.nlayers - 1):
            self.vel_weights[i] = step_size * w_delta[i] + momentum * self.vel_weights[i]
            self.syn[i] -= self.vel_weights[i]
            self.vel_bias[i] = step_size * b_delta[i] + momentum * self.vel_bias[i]
            self.bias[i] -= self.vel_bias[i]


    def adagrad(self, y, step_size = 0.01, smoothing = np.exp(-8)):
        if getattr(self, "weight_total", None) is None:
            self.weight_total = [0 for i in range(self.nlayers - 1)]
        if getattr(self, "bias_total", None) is None:
            self.bias_total = [0 for i in range(self.nlayers - 1)]

        w_delta, b_delta = self.grad(y)
        self.weight_total += np.square(w_delta)
        self.bias_total += np.square(b_delta)

        for i in range(self.nlayers - 1):
            self.syn[i] -= (step_size * w_delta[i]) / np.sqrt(self.weight_total[i] + smoothing)
            self.bias[i] -= (step_size * b_delta[i]) / np.sqrt(self.bias_total[i] + smoothing)


    def adadelta(self, y, discount = 0.9, smoothing = np.exp(-8)):
        if getattr(self, "w_avg", None) is None:
            self.w_avg = [0 for i in range(self.nlayers - 1)]
        if getattr(self, "b_avg", None) is None:
            self.b_avg = [0 for i in range(self.nlayers - 1)]

        w_delta, b_delta = self.grad(y)

        for i in range(self.nlayers - 1):
            w_rms_prev = np.sqrt(self.w_avg[i] + smoothing)
            b_rms_prev = np.sqrt(self.b_avg[i] + smoothing)
            self.w_avg[i] = discount * self.w_avg[i] + (1 - discount) * np.square(w_delta[i])
            self.b_avg[i] = discount * self.b_avg[i] + (1 - discount) * np.square(b_delta[i])
            w_rms_curr = np.sqrt(self.w_avg[i] + smoothing)
            b_rms_curr = np.sqrt(self.b_avg[i] + smoothing)
            self.syn[i] -= (w_rms_prev / w_rms_curr) * w_delta[i]
            self.bias[i] -= (b_rms_prev / b_rms_curr) * b_delta[i]


    def adam(self, y, step_size = 0.01, b1 = 0.9, b2 = 0.999, smoothing = np.exp(-8)):
        if getattr(self, "w_moment_one", None) is None:
            self.w_moment_one = [0 for i in range(self.nlayers - 1)]
        if getattr(self, "b_moment_one", None) is None:
            self.b_moment_one = [0 for i in range(self.nlayers - 1)]
        if getattr(self, "w_moment_two", None) is None:
            self.w_moment_two = [0 for i in range(self.nlayers - 1)]
        if getattr(self, "b_moment_two", None) is None:
            self.b_moment_two = [0 for i in range(self.nlayers - 1)]

        w_delta, b_delta = self.grad(y)
        w_corr_one, b_corr_one, w_corr_two, b_corr_two = tuple([0 for i in range(self.nlayers - 1)] for x in range(4))

        for i in range(self.nlayers - 1):
            self.w_moment_one[i] = b1 * self.w_moment_one[i] + (1 - b1) * w_delta[i]
            self.b_moment_one[i] = b1 * self.b_moment_one[i] + (1 - b1) * b_delta[i]
            self.w_moment_two[i] = b2 * self.w_moment_two[i] + (1 - b2) * (np.square(w_delta[i]))
            self.b_moment_two[i] = b2 * self.b_moment_two[i] + (1 - b2) * (np.square(b_delta[i]))

            w_corr_one[i] = self.w_moment_one[i] / (1 - b1)
            b_corr_one[i] = self.b_moment_one[i] / (1 - b1)
            w_corr_two[i] = self.w_moment_two[i] / (1 - b2)
            b_corr_two[i] = self.b_moment_two[i] / (1 - b2)

            self.syn[i] -= (step_size * w_corr_one[i]) / (np.sqrt(w_corr_two[i]) + smoothing)
            self.bias[i] -= (step_size * b_corr_one[i]) / (np.sqrt(b_corr_two[i]) + smoothing)


    def train(self, x, y, step_size = 0.01, epochs = 1000, seed = 1, method = 'gradient_descent', batch_size = None, batch_ratio = 1, momentum = 0.9, discount = 0.9, b1 = 0.9, b2 = 0.999, smoothing = np.exp(-8), notification_frequency = None):
        self.randomize(seed = seed)

        loc = 0
        if batch_size is None:
            batch_size = int(np.ceil(x.shape[0] * batch_ratio))

        for i in range(epochs):
            if batch_size == x.shape[0]:
                batch_x = x
                batch_y = y
            elif(loc + batch_size < x.shape[0]):
                batch_x = x[loc:(loc + batch_size)]
                batch_y = y[loc:(loc + batch_size)]
                loc += batch_size
            else:
                batch_x = x[loc:(x.shape[0])]
                batch_y = y[loc:(x.shape[0])]

                if batch_x.shape[0] > 0 and batch_y.shape[0] > 0:
                    batch_x = np.append(batch_x, x[0:(loc + batch_size - x.shape[0])], axis = 0)
                    batch_y = np.append(batch_y, y[0:(loc + batch_size - x.shape[0])], axis = 0)
                else:
                    batch_x = x[0:(loc + batch_size - x.shape[0])]
                    batch_y = y[0:(loc + batch_size - x.shape[0])]

                loc += batch_size - x.shape[0]

            self.fprop(batch_x)

            if notification_frequency is not None and i % notification_frequency == 0:
                print("Beginning %sth epoch." % (i))
                print("Total batch error: %s" % (np.sum(self.error(batch_y))))

            if(method == 'gradient_descent'):
                self.gradient_descent(batch_y, step_size)
            elif(method == 'momentum'):
                self.momentum(batch_y, step_size, momentum)
            elif(method == 'adagrad'):
                self.adagrad(batch_y, step_size, smoothing)
            elif(method == 'adadelta'):
                self.adadelta(batch_y, discount, smoothing)
            elif(method == 'adam'):
                self.adam(batch_y, step_size, b1, b2, smoothing)


    def save(self, filename = 'params'):
        f = open(filename, 'wb')
        for item in self.syn:
            pk.dump(item, f, protocol = pk.HIGHEST_PROTOCOL)
        for item in self.bias:
            pk.dump(item, f, protocol = pk.HIGHEST_PROTOCOL)
        f.close()


    def load(self, filename = 'params'):
        f = open(filename, 'rb')
        for i in range(self.nlayers - 1):
            self.syn[i] = pk.load(f)
        for i in range(self.nlayers - 1):
            self.bias[i] = pk.load(f)
        f.close()


    def terminate(self):
        print("Process terminated.")
        del self
