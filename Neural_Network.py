import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import trange

rng = np.random.default_rng()


# ACTIVATION FUNCTIONS
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax_stable(x):
    e = np.exp(x - np.max(x, axis=0))  # normalize exponent to avoid overflow
    s = e / np.sum(e, axis=0)
    return s


# leaky ReLU with a = 0.01
def relu(x):
    rel = np.maximum(0.01 * x, x)
    # rel[rel > 6] = 6
    return rel


# ACTIVATIONS DERIVATIVES
# they work with the output of the layer, instead of the local field u.

# sigmoid derivative
def sigmoid_der(out):
    sig_d = out * (1 - out)
    return sig_d


# LREeLU derivative
def relu_der(out):
    rel_d = np.ones(out.shape) * 0.01
    rel_d[out > 0] = 1
    return rel_d


# softmax derivative
def softmax_stable_der(out):
    return out * (1 - out)


# LOSS FUNCTIONS

def half_sq_error(out, ideal):
    return 0.5 * np.mean(np.square(ideal - out))


def x_entropy(out, ideal):
    am = np.argmax(ideal, axis=0)
    r = range(am.size)
    return -np.mean(np.log(out[am, r] + 1e-8))


# LOSS DERIVATIVES

def half_sq_error_der(out, ideal):
    return ideal - out  # -(out - ideal)


# derivative of Cross-Entropy when used with Softmax
def x_entropy_softmax_der(out, ideal):
    return ideal - out  # -(out - ideal)


# settings of each activation function, so we can assign them dynamically to the network.
activations = {'sigmoid': {'func': sigmoid, 'der': sigmoid_der, 'l_rate': 0.1, 'loss': half_sq_error,
                           'loss_der': half_sq_error_der},
               'relu': {'func': relu, 'der': relu_der, 'l_rate': 0.01, 'loss': half_sq_error,
                        'loss_der': half_sq_error_der},
               'softmax': {'func': softmax_stable, 'der': lambda x: 1, 'l_rate': 0.1, 'loss': x_entropy,
                           'loss_der': x_entropy_softmax_der}}


#
def initialize_weights(func: str, in_n: int, neurons: int):
    """Initialize the weights of a layer depending on the activation function it has."""
    weights = None
    biases = None
    if func in ['sigmoid', 'softmax']:
        a = np.sqrt(6) / np.sqrt(in_n + neurons)
        weights = rng.uniform(-a, a, (neurons, in_n))  # Xavier initialization
        biases = rng.uniform(-a, a, (neurons, 1))
    elif func == 'relu':
        weights = rng.normal(0, np.sqrt(2 / in_n), (neurons, in_n))  # Kaiming He initialization
        biases = rng.normal(0, np.sqrt(2 / in_n), (neurons, 1))
    return weights, biases


# Main component of the Neural Network
class HiddenLayer:
    """  Main component of the Neural Network.

         Holds all the information for each neuron in the layer, in column vectors (numpy arrays).
         For example the 1st row of weights matrix holds all the weights for the 1st neuron.
    """

    def __init__(self, neurons_n, input_size, func, momentum):
        self.input = np.zeros((input_size, 1))
        self.n = neurons_n
        self.func = func
        self.act = activations[func]['func']  # dynamically assign activation function
        self.act_der = activations[func]['der']
        self.l_rate = activations[func]['l_rate']
        self.l_rate0 = self.l_rate
        # weights.shape is (neurons_n, input_size) and bias.shape is (neurons_n,1)
        self.weights, self.bias = initialize_weights(func, input_size, neurons_n)
        self.weights_diff = np.zeros(self.weights.shape)  # stores the previous weight gradient
        self.bias_diff = np.zeros(self.bias.shape)  # stores the previous bias gradient
        self.u = np.zeros((self.n, 1))  # local field
        self.output = np.zeros((self.n, 1))
        self.m = momentum

    def feed(self, x):
        """ Calculate the output of the layer when the input is x.
        x must have shape (features, samples), 1 >= samples >= n """
        self.input = x
        self.u = np.dot(self.weights, self.input) + self.bias
        self.output = self.act(self.u)
        return self.output

    def delta(self, weights: np.ndarray, delta: np.ndarray):
        """"Calculates local gradients of layer.
        weights and delta must be from the next layer
        """
        return self.act_der(self.output) * np.dot(weights.T, delta)

    def update(self, grads_w, grads_b):
        """Updates the weights and biases of layer."""
        dw = self.l_rate * grads_w
        db = self.l_rate * grads_b
        self.weights += self.m * self.weights_diff + dw
        self.bias += self.m * self.bias_diff + db
        self.weights_diff = dw
        self.bias_diff = db


class OutputLayer(HiddenLayer):
    """Extends HiddenLayer.
    Has loss function and calculates the local gradient differently.
    """

    def __init__(self, neurons_n, input_size, func, momentum):
        super(OutputLayer, self).__init__(neurons_n, input_size, func, momentum)
        self.loss = activations[func]['loss']
        self.loss_der = activations[func]['loss_der']

    def delta(self, ideal: np.ndarray, **kwargs):
        """Calculates the local gradient using the chain rule.
        Softmax Case: Since the Cross-Entropy derivative already incorporates softmax, act_der just returns 1.
        """
        return self.loss_der(self.output, ideal) * self.act_der(self.output)


class NNet:
    """Multi-Layer Perceptron. """

    def __init__(self, layout, func: str, momentum=0.9, softmax=True):
        layout = np.asarray(layout)  # make sure layout is np.ndarray
        self.length = layout.size - 1  # number of layers without input layer
        self.layers = []  # the input layer is essentially the input array of the 1st hidden layer
        self.step = 10  # epochs to decrease learning rate
        # initialize hidden layers
        for i in range(1, self.length):
            self.layers.append(HiddenLayer(layout[i], layout[i - 1], func, momentum))
        if softmax:
            func = 'softmax'
            # initialize output layer
        self.layers.append(OutputLayer(layout[-1], layout[-2], func, momentum))

    def decay(self, epoch):
        """Decreases the learning rate based on the epochs."""
        i = 1 / (2**(epoch // self.step))
        for layer in self.layers:
            layer.l_rate = layer.l_rate0 * i
        return

    def feed(self, x):
        """Feed-forward pass of x through the Network."""
        out = x
        # the output of each layer is the input of the next one.
        for layer in self.layers:
            out = layer.feed(out)
        return out

    def update(self, grads, biases, *size):
        """Updates the weights and biases of the Network.
        When its batch update, the size of the batch is passed in *size argument"""
        if len(size) > 0:  # check if its batch update or stochastic update
            for i in range(len(grads)):
                grads[i] = grads[i] / size[0]  # average each gradient
                biases[i] = biases[i].sum(axis=1, keepdims=True) / size[0]
        # update each layer
        for layer, grad, bias in zip(self.layers, grads, biases):
            layer.update(grad, bias)

    def backpropagation(self, y):
        # initialize list of numpy arrays to store gradients for each layer
        grads = [np.zeros(layer.weights.shape) for layer in self.layers]
        grads_bias = [np.zeros(layer.bias.shape) for layer in self.layers]
        layer_out = self.layers[-1]
        # start from output layer
        delta = layer_out.delta(y)  # local gradient
        grads[-1] = np.dot(delta, layer_out.input.T)  # (neurons_n,samples).(samples, input_n) -> (neurons_n,input_n)
        grads_bias[-1] = delta  # gradient of biases with input 1
        # do the same for each hidden layer
        for i in reversed(range(self.length - 1)):
            layer = self.layers[i]
            delta = layer.delta(self.layers[i + 1].weights, delta)
            grads[i] = np.dot(delta, layer.input.T)
            grads_bias[i] = delta

        return grads, grads_bias

    def stochastic(self, x, y, **kwargs):
        """Stochastic mode of training.
        Passes the input one by one in random order and updates the weights and biases after each pass"""
        for i in kwargs['perm']:  # list with indexes of x in random order.
            self.feed(x[:, i:i + 1])
            grads, grads_bias = self.backpropagation(y[:, i:i + 1])
            self.update(grads, grads_bias)

    def batch(self, x, y, **kwargs):
        """ Batch/Mini-Batch mode of training.
        Passes multiple samples of x as input each time, updates the weight and biases after each pass
        """
        # split the array into batch_n sub-arrays of size batch. the last sub-array will have size: samples % batch
        perms = np.array_split(kwargs['perm'], kwargs['batch_n'])
        for perm in perms:
            self.feed(x[:, perm])
            grads, grads_bias = self.backpropagation(y[:, perm])
            self.update(grads, grads_bias, perm.size)

    def train(self, x: np.ndarray, y: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, batch: int, max_epochs: int):
        """Trains the network by adjusting the weights and biases based on (x,y) dataset. """
        # check if it is batch or stochastic training and assign the correct function.
        # self.stochastic is faster than self.batch when batch size is 1.
        if batch > 1:
            mode = self.batch
        else:
            mode = self.stochastic
        batch_n = x.shape[1] // batch  # number of sub-arrays needed based on batch size.
        # calculate average cost for the training dataset, accuracy for both sets. return the in a dictionary
        stats_dict = self.stats(x, y, x_test, y_test)
        cost_list = [tuple(stats_dict.values())]  # store values in a list to return
        t = trange(max_epochs, position=0, postfix=stats_dict)  # create progress bar for terminal
        for epoch in t:
            perm = rng.permutation(x.shape[1])  # create a list of x indexes in random order
            mode(x, y, perm=perm, batch_n=batch_n)  # train the network
            self.decay(epoch)  # calculate the new learning rate
            stats_dict = self.stats(x, y, x_test, y_test)  # collect validation stats
            cost_list.append(tuple(stats_dict.values()))  # store stats
            t.set_postfix(stats_dict)  # update progress bar info
        return cost_list  # return a list with validation stats from each epoch

    def test(self, x, y):
        """"Calculates the average cost of the Network based on (x,y) dataset """
        cost = self.layers[-1].loss(self.feed(x), y)
        return cost

    def predict(self, x, y):
        """Returns the confusion matrix based on the output of the Network. """
        y = np.argmax(y, axis=0)  # Reverse one-hot encoding
        output = self.feed(x)  # get output of network
        output = np.argmax(output, axis=0)  # choose the class with most likelihood as the answer of the network
        cm = confusion_matrix(y_true=y, y_pred=output)  # calculate confusion matrix. (sklearn)
        return cm

    def stats(self, train_x, train_y, test_x, test_y):
        """Helper function for training."""
        out = self.feed(train_x)
        test_out = self.feed(test_x)
        cost = self.layers[-1].loss(out, train_y)
        acc = []
        for out, y in zip([out, test_out], [train_y, test_y]):
            y = np.argmax(y, axis=0)
            out = np.argmax(out, axis=0)
            cm = confusion_matrix(y_true=y, y_pred=out, normalize='all')
            acc.append(100 * cm.trace())

        return {'cost': cost, 'train': acc[0], 'test': acc[1]}
