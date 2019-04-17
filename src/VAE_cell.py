from __future__ import division
import time
import os
import gzip

import numpy as np
import theano
import theano.tensor as T

import cPickle
from collections import OrderedDict

#import importlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab

epsilon = 1e-8

def relu(x):
    return T.switch(x<0, 0, x)


class VAE_cell:
    """This class implements the Variational Auto Encoder"""
    def __init__(self, continuous, hu_encoder, hu_decoder, n_latent, x_train, x_cln, b1=0.05, b2=0.001, batch_size=100, learning_rate=0.000001, lam=0):
        self.continuous = continuous
        self.hu_encoder = hu_encoder
        self.hu_decoder = hu_decoder
        self.n_latent = n_latent
        [self.N, self.features] = x_train.shape

        self.prng = np.random.RandomState(42)

        self.b1 = b1
        self.b2 = b2
        self.learning_rate = learning_rate
        self.lam = lam

        self.batch_size = batch_size

        sigma_init = 0.01

        create_weight = lambda dim_input, dim_output: self.prng.normal(0, sigma_init, (dim_input, dim_output)).astype(theano.config.floatX)
        create_bias = lambda dim_output: np.zeros(dim_output).astype(theano.config.floatX)

        # encoder
        W_xh = theano.shared(create_weight(self.features, hu_encoder), name='W_xh')
        b_xh = theano.shared(create_bias(hu_encoder), name='b_xh')

        W_hmu = theano.shared(create_weight(hu_encoder, n_latent), name='W_hmu')
        b_hmu = theano.shared(create_bias(n_latent), name='b_hmu')

        W_hsigma = theano.shared(create_weight(hu_encoder, n_latent), name='W_hsigma')
        b_hsigma = theano.shared(create_bias(n_latent), name='b_hsigma')

        # decoder
        W_zh = theano.shared(create_weight(n_latent, hu_decoder), name='W_zh')
        b_zh = theano.shared(create_bias(hu_decoder), name='b_zh')

        self.params = OrderedDict([("W_xh", W_xh), ("b_xh", b_xh), ("W_hmu", W_hmu), ("b_hmu", b_hmu),
                                   ("W_hsigma", W_hsigma), ("b_hsigma", b_hsigma), ("W_zh", W_zh),
                                   ("b_zh", b_zh)])
        
        if self.continuous:
            W_hxmu = theano.shared(create_weight(hu_decoder, self.features), name='W_hxmu')
            b_hxmu = theano.shared(create_bias(self.features), name='b_hxmu')

            W_hxsig = theano.shared(create_weight(hu_decoder, self.features), name='W_hxsigma')
            b_hxsig = theano.shared(create_bias(self.features), name='b_hxsigma')

            self.params.update({'W_hxmu': W_hxmu, 'b_hxmu': b_hxmu, 'W_hxsigma': W_hxsig, 'b_hxsigma': b_hxsig})
        else:
            W_hx = theano.shared(create_weight(hu_decoder, self.features), name='W_hx')
            b_hx = theano.shared(create_bias(self.features), name='b_hx')

            self.params.update({'W_hx': W_hx, 'b_hx': b_hx})

        # Adam parameters
        self.m = OrderedDict()
        self.v = OrderedDict()

        for key, value in self.params.items():
                self.m[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='m_' + key)
                self.v[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='v_' + key)

        x_train = theano.shared(x_train.astype(theano.config.floatX), name="x_train")
        x_cln = theano.shared(x_cln.astype(theano.config.floatX), name="x_cln")

        self.update, self.likelihood, self.encode, self.decode = self.create_gradientfunctions(x_train, x_cln)

        #testSample = theano.shared(testSample.astype(theano.config.floatX), name="testSample")
        #self.reconstruct self.create_reconstructionFunc(testSample)


    def encoder(self, x):
        h_encoder = relu(T.dot(x, self.params['W_xh']) + self.params['b_xh'].dimshuffle('x', 0))

        mu = T.dot(h_encoder, self.params['W_hmu']) + self.params['b_hmu'].dimshuffle('x', 0)
        log_sigma = T.dot(h_encoder, self.params['W_hsigma']) + self.params['b_hsigma'].dimshuffle('x', 0)

        return mu, log_sigma

    def sampler(self, mu, log_sigma):
        seed = 42

        if "gpu" in theano.config.device:
            #srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=seed)
            srng = T.shared_randomstreams.RandomStreams(seed=seed)
        else:
            srng = T.shared_randomstreams.RandomStreams(seed=seed)

        eps = srng.normal(mu.shape)

        # Reparametrize
        z = mu + T.exp(0.5 * log_sigma) * eps

        return z

    def decoder(self, x, z, x_c):
        h_decoder = relu(T.dot(z, self.params['W_zh']) + self.params['b_zh'].dimshuffle('x', 0))

        if self.continuous:
            reconstructed_x = T.dot(h_decoder, self.params['W_hxmu']) + self.params['b_hxmu'].dimshuffle('x', 0)
            log_sigma_decoder = T.dot(h_decoder, self.params['W_hxsigma']) + self.params['b_hxsigma']

            logpxz = (-(0.5 * np.log(2 * np.pi) + 0.5 * log_sigma_decoder) -
                      0.5 * ((x_c - reconstructed_x)**2 / T.exp(log_sigma_decoder))).sum(axis=1, keepdims=True)
        else:
            reconstructed_x = T.nnet.sigmoid(T.dot(h_decoder, self.params['W_hx']) + self.params['b_hx'].dimshuffle('x', 0))
            logpxz = - T.nnet.binary_crossentropy(reconstructed_x, x_c).sum(axis=1, keepdims=True)

        return reconstructed_x, logpxz

    def create_gradientfunctions(self, x_train, x_cln):
        """This function takes as input the whole dataset and creates the entire model"""
        x = T.matrix("x")
        x_c = T.matrix("x_c")

        epoch = T.scalar("epoch")

        batch_size = x.shape[0]

        mu, log_sigma = self.encoder(x)
        z = self.sampler(mu, log_sigma)
        reconstructed_x, logpxz = self.decoder(x,z, x_c)

        # Expectation of (logpz - logqz_x) over logqz_x is equal to KLD (see appendix B):
        KLD = 0.5 * T.sum(1 + log_sigma - mu**2 - T.exp(log_sigma), axis=1, keepdims=True)

        # Average over batch dimension
        logpx = T.mean(logpxz + KLD)

        # Compute all the gradients
        gradients = T.grad(logpx, self.params.values())

        # Adam implemented as updates
        updates = self.get_adam_updates(gradients, epoch)

        batch = T.iscalar('batch')

        givens = {
            x: x_train[batch*self.batch_size:(batch+1)*self.batch_size, :],
            x_c: x_cln[batch*self.batch_size:(batch+1)*self.batch_size, :]
        }

        # Define a bunch of functions for convenience
        update = theano.function([batch, epoch], logpx, updates=updates, givens=givens)
        likelihood = theano.function([x, x_c], logpx)
        encode = theano.function([x], z)
        decode = theano.function([z], reconstructed_x)

        return update, likelihood, encode, decode

    def transform_data(self, x_train):
        transformed_x = np.zeros((self.N, self.n_latent))
        batches = np.arange(int(self.N / self.batch_size))

        for batch in batches:
            batch_x = x_train[batch*self.batch_size:(batch+1)*self.batch_size, :]
            transformed_x[batch*self.batch_size:(batch+1)*self.batch_size, :] = self.encode(batch_x)

        return transformed_x

    def save_parameters(self, path):
        """Saves all the parameters in a way they can be retrieved later"""
        cPickle.dump({name: p.get_value() for name, p in self.params.items()}, open(path + "/params.pkl", "wb"))
        cPickle.dump({name: m.get_value() for name, m in self.m.items()}, open(path + "/m.pkl", "wb"))
        cPickle.dump({name: v.get_value() for name, v in self.v.items()}, open(path + "/v.pkl", "wb"))

    def load_parameters(self, path):
        """Load the variables in a shared variable safe way"""
        p_list = cPickle.load(open(path + "/params.pkl", "rb"))
        m_list = cPickle.load(open(path + "/m.pkl", "rb"))
        v_list = cPickle.load(open(path + "/v.pkl", "rb"))

        for name in p_list.keys():
            self.params[name].set_value(p_list[name].astype(theano.config.floatX))
            self.m[name].set_value(m_list[name].astype(theano.config.floatX))
            self.v[name].set_value(v_list[name].astype(theano.config.floatX))

    def get_adam_updates(self, gradients, epoch):
        updates = OrderedDict()
        gamma = T.sqrt(1 - self.b2**epoch) / (1 - self.b1**epoch)

        values_iterable = zip(self.params.keys(), self.params.values(), gradients,
                              self.m.values(), self.v.values())

        for name, parameter, gradient, m, v in values_iterable:
            new_m = self.b1 * m + (1. - self.b1) * gradient
            new_v = self.b2 * v + (1. - self.b2) * (gradient**2)

            updates[parameter] = parameter + self.learning_rate * gamma * new_m / (T.sqrt(new_v) + epsilon)

            if 'W' in name:
                # MAP on weights (same as L2 regularization)
                updates[parameter] -= self.learning_rate * self.lam * (parameter * np.float32(self.batch_size / self.N))

            updates[m] = new_m
            updates[v] = new_v

        return updates

def imagify(flat_image, h, w):
    image = flat_image.reshape([h, w])
    return image / image.max()

#def create_reconstructionFunc(testSample):
    """This function takes one sample as input and reconstruct this sample"""



#def encoder_real(im, param):
#        h_encoder = relu(T.dot(im, param['W_xh']) + param['b_xh'])
#        mu = T.dot(h_encoder, param['W_hmu']) + param['b_hmu']
#        log_sigma = T.dot(h_encoder, param['W_hsigma']) + param['b_hsigma']

        #return mu, log_sigma


#if os.path.isfile(path + "params.pkl"):
#    print "Restarting from earlier saved parameters!"
#    p_list = cPickle.load(open(path + "/params.pkl", "rb"))
#    m_list = cPickle.load(open(path + "/m.pkl", "rb"))
#    v_list = cPickle.load(open(path + "/v.pkl", "rb"))

if __name__ == "__main__":
    continuous = False
    hu_encoder = 800
    hu_decoder = 800
    n_latent = 48

    print "Loading MNIST data"
    # Retrieved from: http://deeplearning.net/data/mnist/mnist.pkl.gz
    #f = gzip.open('mnist.pkl.gz', 'rb')
    f = gzip.open('basic.pkl.gz', 'rb')
    (x_train, t_train), (x_valid, t_valid), (x_test, t_test) = cPickle.load(f)
    f.close()
    x_cln = x_valid
    t_cln = t_valid

    path = "./"

    print "instantiating model"
    model = VAE_cell(continuous, hu_encoder, hu_decoder, n_latent, x_train, x_cln)

    model.load_parameters(path)

    x_ = T.vector('x_')
    x_clean = T.vector('x_clean')
    mu, log_sigma = model.encoder(x_)
    z = model.sampler(mu, log_sigma)
    reconstructed_x, logpxz = model.decoder(x_,z, x_clean)
    reconFunc = theano.function([x_], [reconstructed_x], allow_input_downcast=True)
    """encode_ = theano.function([x_], z)
    decode_ = theano.function([z], reconstructed_x)"""

    for testSample in x_test:

        reconImg1 = reconFunc(testSample)
        reconImg2 = reconFunc(testSample)
        reconImg3 = reconFunc(testSample)
        reconImg_1 = np.asanyarray(reconImg1, dtype=theano.config.floatX)
        #reconImg_2 = np.asanyarray(reconImg2, dtype=theano.config.floatX)
        #reconImg_3 = np.asanyarray(reconImg3, dtype=theano.config.floatX)
        pylab.figure()
        pylab.gray()
        pylab.imshow(imagify(testSample, 28, 28), interpolation='nearest')
        pylab.figure()
        pylab.gray()
        pylab.imshow(imagify(reconImg_1, 28, 28), interpolation='nearest')
        #pylab.figure()
        #pylab.gray()

        #pylab.imshow(imagify(reconImg_2, 28, 28), interpolation='nearest')
        #pylab.figure()
        #pylab.gray()
        #pylab.imshow(imagify(reconImg_3, 28, 28), interpolation='nearest')
        pylab.show()