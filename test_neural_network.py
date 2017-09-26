import cPickle
import os
import gzip
import matplotlib.pyplot as plt
import pylab
import numpy as np

#### Libraries
# Standard library
import gzip
import pylab

# Third-party libraries
import numpy as np

weights=[]
biases=[]
outs=[]
costs=[]

def mvectorized_result(j,classifications):
    """Return a classifications-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
#    e=[j]
    e = np.zeros((classifications, 1))
    e[j] = 1.0
    return e

def mload_data_wrapper(tr_input,tr_output,classification = 0):
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    code."""
    ni=np.array(tr_input)
    no=np.array(tr_output)
    input_units=len(tr_input[0])
    if classification:
        output_units=classification
    else:
        output_units=len(tr_output[0])
    tr_data=(ni,no)
    tr_d, va_d, te_d = (tr_data,tr_data,tr_data)
    training_inputs = [np.reshape(x, (input_units, 1)) for x in tr_d[0]]
    print tr_d[1]
    if classification:
        training_results = [mvectorized_result(y-1,output_units) for y in tr_d[1]]
    else:
        training_results = [np.reshape(y, (output_units,1)) for y in tr_d[1]]
    #training_results = [y for y in tr_d[1]]
#    print "trr type=",type(training_results)
#    print training_results
    training_data = zip(training_inputs, training_results)
    print "training_data=",training_data
    validation_inputs = [np.reshape(x, (input_units, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (input_units, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

#tr_d, va_d, te_d = mload_data()
#img=tr_d[0][99].reshape(28,28)
#print(tr_d[0][3])
#print(tr_d[1][3])
#print type(tr_d[0])
#print type(tr_d[1])
#print type(tr_d[0][0])
#print type(tr_d[1][0])
#pylab.imshow(img)
#pylab.gray()
#pylab.show()
"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y)

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)

class Network(object):

    def __init__(self, sizes, cost=QuadraticCost):
    #def __init__(self, sizes, cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.cost = cost
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        #self.biases[0][0][0] = 0.9
        #self.weights[0][0][0] = 0.6
        self.biases[0][0][0] = 2
        self.weights[0][0][0] = 2
        #print "sizes = ",self.sizes
        print "biases = ",self.biases
        print "weights = ",self.weights

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        print "a=",a
        for b, w in zip(self.biases, self.weights):
            print "b=",b
            print "w=",w
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
#                print "Epoch {0}: {1} / {2}".format(
#                    j, self.evaluate(test_data), n_test)
                self.mevaluate(test_data)
            else:
                print "Epoch {0} complete".format(j)
        print "weights = ",weights
        print "biases = ",biases
        print "costs = ",costs
        print "len=",len(weights)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        weights.append(self.weights[0][0][0])
        biases.append(self.biases[0][0][0])

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        cost = self.cost.fn(activations[-1][0], y[0][0])
        outs.append(activations[-1][0])
        costs.append(cost)
        print "act=",activations[-1][0] , " y=", y[0][0] ," cost=", cost
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def dump(self, x, y):
#        print "x= ",x, " y=",y
        o = self.feedforward(x)
        print "o =",o
        return (np.argmax(o), y)
        
    def mevaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        #test_results = [(np.argmax(self.feedforward(x)), y)
        #                for (x, y) in test_data]
        test_results = [self.dump(x,y) for (x, y) in test_data]
#        print "test_results=",test_results
        return 1

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        #test_results = [(np.argmax(self.feedforward(x)), y)
        #                for (x, y) in test_data]
        test_results = [self.dump(x,y) for (x, y) in test_data]
#        print "test_results=",test_results
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

#print "kkk"
#t=[1,2,3,4,5,6]
#a=np.array(t)#[np.random.randn(y, 1) for y in t[1:]]#]np.array(t)
#print a
#d=[np.reshape(a, (2,3))]
#print d 
##test_data=([[1,2],[3,4],[5,6],[7,8]],[3,7,11,15])
##test_data=([[1,2],
##                  [3,4],
##                  [5,6],
##                  [7,8]], [[3],[7],[11],[15]])
###
#nn = network.Network([2,3,1])
##nn.SGD(test_data,30,1,3.0)
#nn.evaluate(test_data)

#training_data,validation_data,test_data = mload_data_wrapper([[0.05,0.1],[1,3]],[[0.68,0.01],[0.02,0.9]],0)
#training_data,validation_data,test_data = mload_data_wrapper([[0.05,0.1],[1,3]],[[0.01],[0.99]],0)
#training_data,validation_data,test_data = mload_data_wrapper([[0.05,0.1],[1,3]],[1,2],2)
#training_data,validation_data,test_data = mload_data_wrapper([[1,2]],[[.89,0.01]],0)
training_data,validation_data,test_data = mload_data_wrapper([[1]],[[0]],0)
###net = network.Network([2,3,1])
#net = Network([2,3,2])
#net = Network([1,3,1])
net = Network([1,1])
#print net.feedforward(np.array([[1.1,2.2]]))
print net.mevaluate(test_data)
net.SGD(training_data,300,1,0.15,test_data=test_data)

#training_data,validation_data,test_data = mnist_loader.load_data_wrapper()
#net = network.Network([784,30,10])
#net.SGD(test_data,30,1,3.0,test_data=test_data)

fig = plt.figure()
fig.suptitle('stock',fontsize=14,fontweight='bold')
#ax = fig.add_subplot(1,1,1)

#x=np.arange(0,30,.01)
x=[]
y=[]
for i in range(0,300):
    x.append(i)
    y.append(0)
ax = fig.add_subplot(1,1,1)

#plt.plot(x,weights,color='r',linewidth=1.5, linestyle="-", label="weight")
#plt.plot(x,biases,color='g',label="biases")
plt.plot(x,costs,color='b',label="costs")
plt.plot(x,y,color='black',linewidth=0.5, linestyle="-")

plt.legend(loc='upper right')
plt.show();

plt.plot(x,outs,color='r',label="out")
plt.plot(x,y,color='black',linewidth=0.5, linestyle="-")
plt.legend(loc='upper right')
plt.show();

plt.plot(x,weights,color='r',linewidth=1.5, linestyle="-", label="weight")
plt.plot(x,biases,color='g',label="biases")
plt.legend(loc='upper right')
plt.show();
