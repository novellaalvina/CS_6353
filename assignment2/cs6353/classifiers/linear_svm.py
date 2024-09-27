import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on mini batches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a mini batch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

        """L(W) = sum(L_i)/N + lambda * R(W)
           sum(L_i)/N is the data loss part. lambda * R(W) is the regularization part.
           L_i = sum [max(0, w_j * x_i - w_y_i * x_i + delta)]
           R(W) = sum_of_k sum_of_l (W_k,l)^2   -> L2 regularization
        """
        # step 1
        # update the loss gradient (data loss) in respect to the weight based on the classes
        """dL_i/dW_y[i] = -(sum(indicator_1(loss)))x_i"""
        """dL_i/dW_j = (sum(indicator_1(loss)))x_i"""
        dW[:, y[i]] -= X[i] # with respect to the class
        dW[:, j] += X[i]    # with respect to the incorrect class

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  # step 2
  # update the loss gradient (data loss) in respect to the weight for the overall loss gradient
  """dL/dW for the data loss part is the current (dL_i/dW)/N"""
  dW /= len(X)
  # update the loss gradient (regularization) in respect to the weight
  """Let f is the regularization part of the loss function. So df/dW = 2 * lambda * W_k,l.  
     Hence, the overall loss gradient in respect to weight is the sum of the loss gradient of the data loss and regularization part.
     dL/dW = dL_i/dW + df/dW"""
  dW += (2* reg * W)
  
  #####################################################################
  #                       END OF YOUR CODE                            #
  #####################################################################
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  scores_y = scores[range(len(X)), y][:, np.newaxis]
  margins = np.maximum(0, scores - scores_y + 1)
  # margins[y] = 0
  loss = np.sum(margins)/len(X) - 1 + reg * np.sum(W**2)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  dW = np.where(margins > 0, 1, 0)                                        # gradient for incorrect class
  dW[range(len(X)), y] -= np.count_nonzero(dW>0, axis=1, keepdims=False)   # gradient for correct class
  dW = X.T.dot(dW)/len(X) + (2 * reg * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
