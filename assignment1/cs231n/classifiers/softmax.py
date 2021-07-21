from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train): 
        scores = X[i].dot(W)
        exp_scores = np.exp(scores)
        summed = exp_scores.sum()
        loss += -np.log(exp_scores[y[i]]/summed)
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] -= X[i]
            dW[:, j] += (X[i] * exp_scores[j]) / summed 
    
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    dW /= num_train
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    C = W.shape[1]
    
    exp_scores = np.exp(X @ W)
    corresponding_exp_scores = np.choose(y, exp_scores.T)
    summed = exp_scores.sum(axis=1)
    indices = (np.arange(C) == y[:,None]).astype(int)
    
    loss = -np.sum(np.log(corresponding_exp_scores / summed)) / N + reg * np.sum(W * W)
    dW = X.T @ (exp_scores / summed[:, None] - indices) / N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
