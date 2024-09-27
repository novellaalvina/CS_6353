import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  """L_i = -log(exp(s_y_i) / sum(exp(s_j)))
         = -s_y_i + log(sum(e^(s_j)))
      
      s = w * x_i, j = 0, ..., C"""
  
  for i in range(len(X)):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    score_correct_class = scores[y[i]]

    """dL_i/ds_j = 
      if j == y_i:
        -1 + (1/sum(e^s_j)) * e^(s_j)
      if j != y_i:
        (1/sum(e^s_j)) * e^(s_j)
        
      dL_i/dW = dL_i/ds_j * ds_j/dW
      if j == y_i:
        (-1 + (1/sum(e^s_j)) * e^(s_j)) * x_i
      if j != y_i:
        ((1/sum(e^s_j)) * e^(s_j)) * x_i"""
    
    for j in range(len(scores)):
      f = np.exp(scores[j])/np.sum(np.exp(scores))
      if j == y[i]:
        f -= 1
        dW[:, y[i]] += f * X[i]
      else:
        dW[:, j] += f * X[i]

    loss += -score_correct_class + np.log(np.sum(np.exp(scores)))

  """ Right now the loss is a sum over all training examples, but we want it
    to be an average instead so we divide by num_train. This is L_i."""
  loss /= len(X)
  dW /= len(X)

  """L = L_i + regularization"""
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2* reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores -= np.max(scores, axis=1, keepdims=True)
  p = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
  loss_i = -np.log(p[range(len(X)), y])
  loss = np.sum(loss_i)/len(X) + (reg * np.sum(W*W))

  p[range(len(X)), y] -= 1                    # gradient for correct class
  dW = X.T.dot(p)/len(X) + (2 * reg * W)      # gradient for incorrect class

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW