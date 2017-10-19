#!/bin/env python
# -*- coding: utf-8 -*-

import cPickle
import os
import gzip
import matplotlib.pyplot as plt
import mnist_loader
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
from data_operation import *

#### Define the quadratic and cross-entropy cost functions

class CrossEntropyCost(object):

    #@staticmethod
    #def loss(y, p):
    #    # Avoid division by zero
    #    p = np.clip(p, 1e-15, 1 - 1e-15)
    #    return - y * np.log(p) - (1 - y) * np.log(1 - p)

    @staticmethod
    def loss(a, y):

        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y): # only apply for sigmoid activation function
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y)

    @staticmethod
    def gradient(y, p):
        # Avoid division by zero
        #p = np.clip(p, 1e-15, 1 - 1e-15)
        #return - (y / p) + (1 - y) / (1 - p)
        return (y-p) / (y*(1-y))

    @staticmethod
    def acc(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

class QuadraticCost(object):

    @staticmethod
    def loss(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y): # only apply for sigmoid activation function
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)

    @staticmethod
    def gradient(a, y):
        return -(y-a)

    @staticmethod
    def acc(self, y, p):
        return 0

"""network2.py
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.

"""

#### Libraries
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np
from layers import *
from activation_functions import *

#### Main Network class
class Network(object):

    #def __init__(self, sizes, cost = CrossEntropyCost, activation_fn = Activation('sigmoid')):
    def __init__(self, sizes, cost = QuadraticCost, activation_fn = Activation('sigmoid')):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).

        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost_fn = cost
        self.activation_fn = activation_fn
        #self.biases[0][0][0] = 1
        #self.weights[0][0][0] = 1
        #self.weights[0][0][1] = 1
        #self.biases[0][0][0] = 2
        #self.weights[0][0][0] = 2
        print "biases=",self.biases
        print "weights=",self.weights

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
#            a = sigmoid(np.dot(w, a)+b)
           a = self.activation_fn.function(np.dot(w, a)+b)
#        print "out = ",a
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data = None,
            monitor_training_cost = True,
            monitor_training_accuracy = False,
            monitor_evaluation_cost = False,
            monitor_evaluation_accuracy = False):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.
        """

        if evaluation_data: n_evaluation_data = len(evaluation_data)
        if evaluation_data is None: n_evaluation_data = 0
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))

            print "Epoch %s training complete" % j
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print "Cost on training data: {}".format(cost)
            if monitor_training_accuracy:
                accuracy = self.classification_accuracy(training_data, convert=True)
                training_accuracy.append(accuracy*1.0/n)
                print "Accuracy on training data: {} / {}".format(accuracy, n)
            if monitor_evaluation_cost and n_evaluation_data > 0:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print "Cost on evaluation data: {}".format(cost)
            if monitor_evaluation_accuracy and n_evaluation_data > 0:
                accuracy = self.classification_accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy*1.0/n_evaluation_data)
                print "Accuracy on evaluation data: {} / {}".format(
                    self.classification_accuracy(evaluation_data), n_evaluation_data)
        return training_cost, training_accuracy, evaluation_cost, evaluation_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.
        """
        #print "len(mini_batch=",len(mini_batch)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

        weights.append(self.weights[0][0][0])
        biases.append(self.biases[0][0][0])


    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
#            activation = sigmoid(z)
            activation = self.activation_fn.function(z)
            activations.append(activation)

        # 计算激活函数(activation_function)的导数
        derivatives = self.activation_fn.derivatives(zs[-1])
        # 计算损失函数(cost_function)的梯度
        gradient = self.cost_fn.gradient(activations[-1], y)
        # 计算偏差
        # backward pass
        delta = derivatives * gradient
        #print "zs[-1]=",zs[-1]
        #print "activations[-1]=",activations[-1]
        print "derivatives=",derivatives
        print "grad=",gradient
        print "delta=",delta

        # backward pass
#        delta = (self.cost_fn).delta(zs[-1], activations[-1], y)

        cost = self.cost_fn.loss(activations[-1][0], y[0][0])
        outs.append(activations[-1][0])

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
#            sp = sigmoid_prime(z)
            sp = self.activation_fn.derivatives(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def classification_accuracy(self, data, convert = False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.
        """

        if convert: # for training data
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:       # for validation or test data
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert = False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            print "x=",x,",y=",y
            a = self.feedforward(x)
            print "out=",a
            if convert: y = vectorized_result(y) # for validation or test data
            cost += self.cost_fn.loss(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost_fn": str(self.cost_fn.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost = cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

epoch = 330
## ***************************************** #
## case1
training_data=[(np.array([[1.12],[2.8]]),   np.array([[0.89]]))]
#training_data=[(np.array([[1],[2],[3]]),   np.array([[0.3],[0.09],[0.18]]))]
test_data=training_data
####net = network.Network([2,3,1])
#net = Network([3,3,9,9,3])
net = Network([2,1],activation_fn = Activation('sigmoid'))
#print net.mevaluate(test_data)
net.SGD(training_data,epoch,5,3.0)
e = net.feedforward([[1.12],[2.8]])
print "o = ", e, " argmax=",np.argmax(e)
### ***************************************** #
### case1
#training_data=[(np.array([[1.12],[2.8]]),   np.array([[0],[1]]))]
##training_data=[(np.array([[1],[2],[3]]),   np.array([[0.3],[0.09],[0.18]]))]
#test_data=training_data
#####net = network.Network([2,3,1])
##net = Network([3,3,9,9,3])
#net = Network([2,3,2])
##print net.mevaluate(test_data)
#net.SGD(training_data,epoch,5,3.0)
#e = net.feedforward([[1.13],[2.78]])
#print "o = ", e, " argmax=",np.argmax(e)
#
## ***************************************** #
## case2
#training_data=[(np.array([[1]]),   np.array([[0]]))]
##training_data=[(np.array([[1],[2],[3]]),   np.array([[0.3],[0.09],[0.18]]))]
#test_data=training_data
#####net = network.Network([2,3,1])
##net = Network([3,3,9,9,3])
#net = Network([1,3,3,1])
##print net.mevaluate(test_data)
#net.SGD(training_data,epoch,5,3.0)
#print "o = ", net.feedforward([[1]])

# ***************************************** #
## case3 
#training_data=[(np.array([[1]]), np.array([[0]]))]
#test_data=training_data
#test_data=None
#
#net = Network([1,1])
#tc,ta,ec,ea=net.SGD(training_data,epoch,1,0.15,
#                    evaluation_data = test_data,
#                    #lmbda=0.1,
#                    monitor_training_cost = True,
#                    monitor_training_accuracy = True,
#                    monitor_evaluation_cost = True,
#                    monitor_evaluation_accuracy = True)
#outss = outs
#outs = []
#net.large_weight_initializer()
#tcl,tal,ecl,eal=net.SGD(training_data,epoch,1,0.15,
#                    evaluation_data=test_data,
#                    #lmbda=0.1,
#                    monitor_training_cost=True,
#                    monitor_training_accuracy=True,
#                    monitor_evaluation_cost=True,
#                    monitor_evaluation_accuracy=True)
#outsl = outs
#
net.save('ddd')
#
####### matplotlab #########
#fig = plt.figure()
#fig.suptitle('stock',fontsize=14,fontweight='bold')
##ax = fig.add_subplot(1,1,1)
#
##x=np.arange(0,30,.01)
#x=[]
#y=[]
#for i in range(0,epoch):
#    x.append(i)
#    y.append(0)
#ax = fig.add_subplot(1,1,1)
#
##plt.plot(x,weights,color='r',linewidth=1.5, linestyle="-", label="weight")
##plt.plot(x,biases,color='g',label="biases")
#print "lentc=",len(tc)
#print tc
#if len(tc) > 0:
#    plt.plot(x,tc,color='r',label="tc")
#if len(ec) >0:
#    plt.plot(x,ec,color='b',label="ec")
#plt.plot(x,y,color='black',linewidth=0.5, linestyle="-")
##
#plt.legend(loc='upper right')
#plt.show();
#
#if len(ta) > 0:
#    plt.plot(x,ta,color='g',label="ta")
#if len(ea) > 0:
#    plt.plot(x,ea,color='y',label="ea")
#plt.plot(x,y,color='black',linewidth=0.5, linestyle="-")
#plt.legend(loc='upper right')
#plt.show();
#
#print "sss=",outss
#print "lll=",outsl
#plt.plot(x,outss,color='r',label="outs")
#plt.plot(x,outsl,color='g',label="outl")
#plt.plot(x,y,color='black',linewidth=0.5, linestyle="-")
#plt.legend(loc='upper right')
#plt.show();
#
##plt.plot(x,weights,color='r',linewidth=1.5, linestyle="-", label="weight")
##plt.plot(x,biases,color='g',label="biases")
##plt.legend(loc='upper right')
##plt.show();
