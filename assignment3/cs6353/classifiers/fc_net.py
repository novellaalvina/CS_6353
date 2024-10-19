from builtins import range
from builtins import object
import numpy as np

from cs6353.layers import *
from cs6353.layer_utils import *


class TwoLayerNet(object):
    '''
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    '''

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        '''
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        '''
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        '''
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        '''
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        w1 = self.params['W1']
        w2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']
        reg = self.reg

        f1, relu_cache = affine_relu_forward(X, w1, b1)    # f1 = np.maximum(0, s1) = np.maximum(0, w1*X + b1)
        scores, score_cache = affine_forward(f1, w2, b2)   # scores = np.dot(f1, w2) + b2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # softmax_prob = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
        
        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * reg * (np.sum(w2*w2) + np.sum(w1*w1))

        dx2, dw2, db2 = affine_backward(dout, score_cache)
        dx1, dw1, db1 = affine_relu_backward(dx2, relu_cache)

        dw2 += (2 * 0.5 * reg * w2)
        dw1 += (2 * 0.5 * reg * w1)

        grads['W1'] = dw1
        grads['W2'] = dw2
        grads['b1'] = db1
        grads['b2'] = db2
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    '''
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    batch/layer normalization as an option. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu} x (L - 1) - affine - softmax

    where batch/layer normalization is optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    '''

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                  normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        '''
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.

        '''
        self.normalization = normalization
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        
        dimensions = [input_dim] + hidden_dims + [num_classes]

        for l in range(len(dimensions)-1):
            self.params[f'W{l+1}'] = np.random.normal(0, weight_scale, (dimensions[l], dimensions[l+1]))
            self.params[f'b{l+1}'] = np.zeros(dimensions[l+1])

            if self.normalization == 'batchnorm':
                self.params[f'gamma{l+1}'] = np.ones(self.params[f'W{l+1}'].shape) # scale 
                self.params[f'beta{l+1}'] = np.zeros(self.params[f'b{l+1}'].shape)  # shift

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        '''
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        '''
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params param since it
        # behaves differently during training and testing.
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
            # {affine - [batch/layer norm] - relu} x (L - 1) - affine - softmax
       
        reg = self.reg
        w = {}
        b = {}
        g = {}
        bt = {}
        l_o = {} # layer outs
        l_c = {}
        l_o[0] = X

        for l in range(self.num_layers):
          cache = []
          w[l] = self.params[f'W{l+1}']
          b[l] = self.params[f'b{l+1}']
          if self.normalization == 'batchnorm':
            g[l] = self.params[f'gamma{l+1}']
            bt[l] = self.params[f'beta{l+1}']

          # affine
          l_o[l], affine_cache = affine_forward(l_o[l], w[l], b[l])    # affine = np.maximum(0, w1*X + b1)
          cache.append(affine_cache)
          
          # batch normalization 1
          if self.normalization == 'batchnorm':
            g[l] = np.sqrt(np.var(l_o[l], keepdims=True))         # mini batch variance 
            bt[l] = np.mean(l_o[l], keepdims=True)                # mini batch mean
            l_o[l] = (l_o[l] - bt[l]) / g[l]                        # normalize
            y1 = np.dot(l_o[l], g[l]) + bt[l]                     # scale and shift

          # relu 1
          l_o[l+1], relu_cache = relu_forward(l_o[l])                 # relu = np.maximum(0, normalized_x1) 
          cache.append(relu_cache)
          
          l_c[l] = cache
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads