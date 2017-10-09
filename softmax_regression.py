# coding:utf8
import numpy as np
import gzip
import cPickle
import theano
import os
import theano.tensor as T
from theano.tensor.nnet import softmax
from theano.tensor.nnet import sigmoid

class SoftMax:
    def __init__(self,MAXT=10000,step=0.15,landa=0):
        self.MAXT = MAXT
        self.step = step
        self.landa = landa  #在此权重衰减项未能提升正确率
        
    def load_theta(self,datapath):
        self.theta = cPickle.load(open(datapath,'rb'))

    def process_train(self,data,label,typenum,batch_size=500):
        valuenum=data.shape[1]
        batches =  data.shape[0] / batch_size
        data = theano.shared(np.asarray(data,dtype=theano.config.floatX))
        label = T.cast(theano.shared(np.asarray(label,dtype=theano.config.floatX)), 'int32')
        x = T.matrix('x')
        y = T.ivector('y')
        index = T.lscalar()
        theta =  theano.shared(value=0.001*np.zeros((valuenum,typenum),
                               dtype=theano.config.floatX),
                               name='theta',borrow=True)
        hx=T.nnet.softmax(T.dot(x,theta))
        #hx=T.nnet.sigmoid(T.dot(x,theta))
        #权重衰减项
        cost =  -T.mean(T.log(hx)[T.arange(y.shape[0]), y]) +0.5*self.landa*T.sum(theta ** 2) 
        g_theta = T.grad(cost, theta)
        updates = [(theta, theta - self.step * g_theta)]
        train_model = theano.function(
                          inputs=[index],
                          outputs=[hx,cost],
                          updates=updates,
                          givens={
                                   x: data[index * batch_size: (index + 1) * batch_size],
                                   y: label[index * batch_size: (index + 1) * batch_size]
                                 },
                          allow_input_downcast=True
                          )
        lastcostJ = np.inf
        stop = False
        epoch = 0
        costj=[]
        while (epoch < self.MAXT) and (not stop):
            epoch = epoch + 1
            for minibatch_index in xrange(batches):
                ot,cost=train_model(minibatch_index)
        #        costj.append(train_model(minibatch_index))
                costj.append(cost)
#                print "len(ot)=",len(ot)
#                print "ot=",np.argmax(ot)
            if np.mean(costj)>=lastcostJ:
                print "costJ is increasing !!!"
                stop=True
            else:
                lastcostJ=np.mean(costj)
                print(( 'epoch %i, minibatch %i/%i,averange cost is %f') %
                        (epoch,minibatch_index + 1,batches,lastcostJ))
        self.theta=theta
        if not os.path.exists('data/softmax.pkl'):
            f= open("data/softmax.pkl",'wb')
            cPickle.dump(self.theta.get_value(),f)
            f.close()
        return self.theta.get_value()

    def process_test(self,data,label,batch_size=500):
        batches = label.shape[0] / batch_size
        data = theano.shared(np.asarray(data,dtype=theano.config.floatX))
        label = T.cast(theano.shared(np.asarray(label,dtype=theano.config.floatX)), 'int32')
        x = T.matrix('x')
        y = T.ivector('y')
        index = T.lscalar()
        hx=T.nnet.softmax(T.dot(x,self.theta))
        predict = T.argmax(hx, axis=1)
        errors=T.mean(T.neq(predict, y))
        test_model = theano.function(
                         inputs=[index],
                         outputs=errors,
                         givens={
                                  x: data[index * batch_size: (index + 1) * batch_size],
                                  y: label[index * batch_size: (index + 1) * batch_size]
                                },
                         allow_input_downcast=True
        )
        test_losses=[]
        for minibatch_index in xrange(batches):
            test_losses.append(test_model(minibatch_index))
        test_score = np.mean(test_losses)
        print(( 'minibatch %i/%i, test error of model %f %%') %
              (minibatch_index + 1,batches,test_score * 100.))

    def h(self,x):
        m = np.exp(np.dot(x,self.theta))
        sump = np.sum(m,axis=1)
        return m/sump

    def predict(self,x):
        return np.argmax(self.h(x),axis=1)

if __name__ == '__main__':
    d=np.random.randn(3,1)
    print d
    x=T.dvector('x')
    #x=T.dmatrix('x')
    y1=sigmoid(x)
    y2=softmax(x)
    f1=theano.function([x],y1)
    f2=theano.function([x],y2)
    print f1
    print f2
    print f1([-1,0,2,3])
    print f2([-1,0,2,3])
    print np.argmax([-1,9,2,3,1])

    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    training_inputs = [np.reshape(x, 784) for x in training_data[0]]
    data = np.array(training_inputs)
    training_inputs = [np.reshape(x, 784) for x in validation_data[0]]
    vdata = np.array(training_inputs)
    f.close()
    softmax = SoftMax()
    print "training_data[1]=",training_data[1]
    print "len(traning_data[1])=",len(training_data[1])
    softmax.process_train(data,training_data[1],10)
    softmax.process_test(vdata,validation_data[1])
    #minibatch 20/20, test error of model 7.530000 %
