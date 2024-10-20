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


# class FullyConnectedNet(object):
#     '''
#     A fully-connected neural network with an arbitrary number of hidden layers,
#     ReLU nonlinearities, and a softmax loss function. This will also implement
#     batch/layer normalization as an option. For a network with L layers,
#     the architecture will be

#     {affine - [batch/layer norm] - relu} x (L - 1) - affine - softmax

#     where batch/layer normalization is optional, and the {...} block is
#     repeated L - 1 times.

#     Similar to the TwoLayerNet above, learnable parameters are stored in the
#     self.params dictionary and will be learned using the Solver class.
#     '''

#     def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
#                   normalization=None, reg=0.0,
#                  weight_scale=1e-2, dtype=np.float32, seed=None):
#         '''
#         Initialize a new FullyConnectedNet.

#         Inputs:
#         - hidden_dims: A list of integers giving the size of each hidden layer.
#         - input_dim: An integer giving the size of the input.
#         - num_classes: An integer giving the number of classes to classify.
#         - normalization: What type of normalization the network should use. Valid values
#           are "batchnorm", "layernorm", or None for no normalization (the default).
#         - reg: Scalar giving L2 regularization strength.
#         - weight_scale: Scalar giving the standard deviation for random
#           initialization of the weights.
#         - dtype: A numpy datatype object; all computations will be performed using
#           this datatype. float32 is faster but less accurate, so you should use
#           float64 for numeric gradient checking.

#         '''
#         self.normalization = normalization
#         self.reg = reg
#         self.num_layers = 1 + len(hidden_dims)
#         self.dtype = dtype
#         self.params = {}

#         ############################################################################
#         # TODO: Initialize the parameters of the network, storing all values in    #
#         # the self.params dictionary. Store weights and biases for the first layer #
#         # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
#         # initialized from a normal distribution centered at 0 with standard       #
#         # deviation equal to weight_scale. Biases should be initialized to zero.   #
#         #                                                                          #
#         # When using batch normalization, store scale and shift parameters for the #
#         # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
#         # beta2, etc. Scale parameters should be initialized to ones and shift     #
#         # parameters should be initialized to zeros.                               #
#         ############################################################################
#         dimensions = [input_dim] + hidden_dims + [num_classes]

#         for l in range(len(dimensions)-1):
#             self.params[f'W{l+1}'] = np.random.normal(0, weight_scale, (dimensions[l], dimensions[l+1]))
#             self.params[f'b{l+1}'] = np.zeros(dimensions[l+1])

#             if self.normalization == 'batchnorm':
#                 self.params[f'gamma{l+1}'] = np.ones(dimensions[l+1]) # scale 
#                 self.params[f'beta{l+1}'] = np.zeros(dimensions[l+1])  # shift
                
#         ############################################################################
#         #                             END OF YOUR CODE                             #
#         ############################################################################

#         # With batch normalization we need to keep track of running means and
#         # variances, so we need to pass a special bn_param object to each batch
#         # normalization layer. You should pass self.bn_params[0] to the forward pass
#         # of the first batch normalization layer, self.bn_params[1] to the forward
#         # pass of the second batch normalization layer, etc.
#         self.bn_params = []
#         if self.normalization=='batchnorm':
#             self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
#         if self.normalization=='layernorm':
#             self.bn_params = [{} for i in range(self.num_layers - 1)]

#         # Cast all parameters to the correct datatype
#         for k, v in self.params.items():
#             self.params[k] = v.astype(dtype)


#     def loss(self, X, y=None):
#         '''
#         Compute loss and gradient for the fully-connected net.

#         Input / output: Same as TwoLayerNet above.
#         '''
#         X = X.astype(self.dtype)
#         mode = 'test' if y is None else 'train'

#         # Set train/test mode for batchnorm params param since it
#         # behaves differently during training and testing.
#         if self.normalization=='batchnorm':
#             for bn_param in self.bn_params:
#                 bn_param['mode'] = mode
#         scores = None
#         ############################################################################
#         # TODO: Implement the forward pass for the fully-connected net, computing  #
#         # the class scores for X and storing them in the scores variable.          #
#         #                                                                          #
#         # When using batch normalization, you'll need to pass self.bn_params[0] to #
#         # the forward pass for the first batch normalization layer, pass           #
#         # self.bn_params[1] to the forward pass for the second batch normalization #
#         # layer, etc.                                                              #
#         ############################################################################       
#         # reg = self.reg
#         # w = {}
#         # b = {}
#         # g = {}
#         # bt = {}
#         # l_c = {}
#         # scores = X

#         # for l in range(self.num_layers-1):
#         #   cache = []
#         #   w[l] = self.params[f'W{l+1}']
#         #   b[l] = self.params[f'b{l+1}']
#         #   if self.normalization == 'batchnorm':
#         #     g[l] = self.params[f'gamma{l+1}']
#         #     bt[l] = self.params[f'beta{l+1}']
          
#         #   # affine
#         #   scores, affine_cache = affine_forward(scores, w[l], b[l])    # affine = np.maximum(0, w1*X + b1)
#         #   cache.append(affine_cache)

#         #   # batch normalization 1
#         #   if self.normalization == 'batchnorm':
#         #     scores, batch_cache = batchnorm_forward(scores, g[l], bt[l], self.bn_params[l])
#         #     cache.append(batch_cache)
            
#         #   # relu 1
#         #   scores, relu_cache = relu_forward(scores)                 # relu = np.maximum(0, normalized_x1) 
#         #   cache.append(relu_cache)
#         #   l_c[l] = cache

#         # w[self.num_layers-1] = self.params[f'W{self.num_layers}']
#         # b[self.num_layers-1] = self.params[f'b{self.num_layers}']
#         # scores, score_cache = affine_forward(scores, w[self.num_layers-1], b[self.num_layers-1])
#         _cache_layer = {}
#         _cache_relu  = {}
#         _cache_dropout = {}
#         _cache_batchnorm = {}

#         scores = X

#         # scores, _cache_layer[0] = affine_forward(X, self.params[('W',0)], self.params[('b',0)])
#         # scores, _cache_relu[0]  = relu_forward(scores)
#         # if self.use_dropout:
#         #   scores, _cache_dropout[0] = dropout_forward(scores, self.dropout_param)

#         #{affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
#         for layer in range(0,self.num_layers - 1):
#           scores, _cache_layer[layer]       = affine_forward(scores, self.params[f'W{layer+1}'], self.params[f'b{layer+1}'])
#           if self.normalization == 'batchnorm':
#             scores, _cache_batchnorm[layer] = batchnorm_forward(scores, self.params[f'gamma{layer+1}'], self.params[f'beta{layer+1}'], self.bn_params[layer])

#           scores, _cache_relu[layer]        = relu_forward(scores)
         
#         #LAST LAYER
#         scores, _cache_layer[self.num_layers-1] = affine_forward(scores, self.params[f'W{self.num_layers-1}'], self.params[f'b{self.num_layers-1}'])
#        ############################################################################
#         #                             END OF YOUR CODE                             #
#         ############################################################################

#         # If test mode return early
#         if mode == 'test':
#             return scores

#         loss, grads = 0.0, {}
#         ############################################################################
#         # TODO: Implement the backward pass for the fully-connected net. Store the #
#         # loss in the loss variable and gradients in the grads dictionary. Compute #
#         # data loss using softmax, and make sure that grads[k] holds the gradients #
#         # for self.params[k]. Don't forget to add L2 regularization!               #
#         #                                                                          #
#         # When using batch/layer normalization, you don't need to regularize the scale   #
#         # and shift parameters.                                                    #
#         #                                                                          #
#         # NOTE: To ensure that your implementation matches ours and you pass the   #
#         # automated tests, make sure that your L2 regularization includes a factor #
#         # of 0.5 to simplify the expression for the gradient.                      #
#         ############################################################################
#                   # {affine - [batch/layer norm] - relu} x (L - 1) - affine - softmax
                  
#         # dw = {}
#         # db = {}
        
#         # loss, dout = softmax_loss(scores, y)
#         # dout, dw[self.num_layers-1], db[self.num_layers-1] = affine_backward(dout, score_cache)
        
#         # for l in range(self.num_layers-2, -1, -1):
#         #   print('l', l)
#         #   dout = relu_backward(dout, l_c[l][-1])
          
#         #   if self.normalization == 'batchnorm':
#         #     dout, dgamma, dbeta = batchnorm_backward(dout, l_c[l][1])
#         #     print(f'gamma{l+1}')
#         #     grads[f'gamma{l+1}'] = dgamma
#         #     grads[f'beta{l+1}'] = dbeta
            
#         #   dout, dw[l], db[l] = affine_backward(dout, l_c[l][0])
        
#         # for key in dw:
#         #   loss += 0.5 * reg * (np.sum(w[key]*w[key]))
#         #   dw[key] += (2 * 0.5 * reg * w[key])
#         #   grads[f'W{key+1}'] = dw[key]
#         #   grads[f'b{key+1}'] = db[key]
#         loss, softmax_dx = softmax_loss(scores, y)
#         dLayer = softmax_dx

#         last_layer = self.num_layers - 1
#         dLayer, grads[f'W{last_layer}'], grads[f'b{last_layer}'] = affine_backward(softmax_dx, _cache_layer[last_layer])

#         for layer in reversed(range(0, self.num_layers - 1)):
         
#           dLayer = relu_backward(dLayer, _cache_relu[layer])
#           if self.normalization == 'batchnorm':
#             dLayer, grads[f'gamma{layer}'], grads[(f'beta{layer}')] = batchnorm_backward(dLayer, _cache_batchnorm[layer])
#           dLayer,  grads[f'W{layer}'], grads[f'b{layer}'] = affine_backward(dLayer, _cache_layer[layer])

#         #scores and softmax_dx have the same shape
#         # for layer in reversed(range(1,self.num_layers - 1)):
#         #   if self.use_batchnorm and layer != self.num_layers - 2:
#         #     dLayer, grads[('dgamma',layer)], grads[('dbeta',layer)] = batchnorm_backward(dLayer, _cache_batchnorm[layer])
#         #   dLayer, grads[('W',layer)], grads[('b',layer)] = affine_backward(dLayer,  _cache_layer[layer])
#         #   if self.use_dropout:
#         #     dLayer = dropout_backward(dLayer, _cache_dropout[layer])
#         #   dLayer = relu_backward(dLayer,   _cache_relu[layer])
        
#         # if self.use_batchnorm:
#         #   dLayer, grads[('dgamma',0)], grads[('dbeta',0)] = batchnorm_backward(dLayer, _cache_batchnorm[0])
#         # dLayer,  grads[('W',0)], grads[('b',0)] = affine_backward(dLayer, _cache_layer[0])
#         #

#         #add regularization
#         for layer in range(self.num_layers):
#             loss += 0.5 * self.reg * np.sum(self.params[f'W{layer+1}']*self.params[f'W{layer+1}'])
#             grads[f'W{layer+1}'] += self.reg * self.params[f'W{layer+1}']
#         ############################################################################
#         #                             END OF YOUR CODE                             #
#         ############################################################################

#         return loss, grads

class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, normalization='batchnorm', reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = True
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################


    for layer, hdim in enumerate(hidden_dims):
      self.params[('b',layer)] = np.zeros(hdim)
      if self.use_batchnorm:
        self.params[("dgamma",layer)] = np.ones(hdim)
        self.params[("dbeta", layer)] = np.zeros(hdim)
      if layer == 0:
        self.params[('W',layer)] = np.random.normal(0, weight_scale, (input_dim, hdim))
      else:
        self.params[('W',layer)] = np.random.normal(0, weight_scale, (hidden_dims[layer - 1], hdim))

    self.params[('W', len(hidden_dims))] = np.random.normal(0, weight_scale, (hidden_dims[len(hidden_dims) - 1], num_classes))
    self.params[('b', len(hidden_dims))] = np.random.normal(0, weight_scale, num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    _cache_layer = {}
    _cache_relu  = {}
    _cache_dropout = {}
    _cache_batchnorm = {}

    scores = X

    # scores, _cache_layer[0] = affine_forward(X, self.params[('W',0)], self.params[('b',0)])
    # scores, _cache_relu[0]  = relu_forward(scores)
    # if self.use_dropout:
    #   scores, _cache_dropout[0] = dropout_forward(scores, self.dropout_param)

    #{affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
    for layer in range(0,self.num_layers - 1):
      scores, _cache_layer[layer]       = affine_forward(scores, self.params[('W',layer)], self.params[('b',layer)])
      if self.use_batchnorm:
        scores, _cache_batchnorm[layer] = batchnorm_forward(scores, self.params[("dgamma",layer)], self.params[("dbeta",layer)], self.bn_params[layer])

      scores, _cache_relu[layer]        = relu_forward(scores)
      
    #LAST LAYER
    scores, _cache_layer[self.num_layers-1] = affine_forward(scores, self.params[('W',self.num_layers-1)], self.params[('b',self.num_layers-1)])
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
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, softmax_dx = softmax_loss(scores, y)
    dLayer = softmax_dx

    last_layer = self.num_layers - 1
    dLayer, grads[('W',last_layer)], grads[('b',last_layer)] = affine_backward(softmax_dx, _cache_layer[last_layer])

    for layer in reversed(range(0, self.num_layers - 1)):
      
      dLayer = relu_backward(dLayer, _cache_relu[layer])
      if self.use_batchnorm:
        dLayer, grads[('dgamma',layer)], grads[('dbeta',layer)] = batchnorm_backward(dLayer, _cache_batchnorm[layer])
      dLayer,  grads[('W',layer)], grads[('b',layer)] = affine_backward(dLayer, _cache_layer[layer])

    #scores and softmax_dx have the same shape
    # for layer in reversed(range(1,self.num_layers - 1)):
    #   if self.use_batchnorm and layer != self.num_layers - 2:
    #     dLayer, grads[('dgamma',layer)], grads[('dbeta',layer)] = batchnorm_backward(dLayer, _cache_batchnorm[layer])
    #   dLayer, grads[('W',layer)], grads[('b',layer)] = affine_backward(dLayer,  _cache_layer[layer])
    #   if self.use_dropout:
    #     dLayer = dropout_backward(dLayer, _cache_dropout[layer])
    #   dLayer = relu_backward(dLayer,   _cache_relu[layer])
    
    # if self.use_batchnorm:
    #   dLayer, grads[('dgamma',0)], grads[('dbeta',0)] = batchnorm_backward(dLayer, _cache_batchnorm[0])
    # dLayer,  grads[('W',0)], grads[('b',0)] = affine_backward(dLayer, _cache_layer[0])
    #

    #add regularization
    for layer in range(self.num_layers):
        loss += 0.5 * self.reg * np.sum(self.params[('W',layer)]*self.params[('W',layer)])
        grads[('W',layer)] += self.reg * self.params[('W',layer)]

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads