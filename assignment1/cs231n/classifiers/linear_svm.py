from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def L_i_vectorized(X, y, W):
    scores = X.dot(W)
    margins = np.maximum(0, scores - scores[y] + 1)
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    
    num_dim = W.shape[0]
    h = 0.0001
    
    for i in range(num_train):
        curr_loss = L_i_vectorized(X[i], y[i], W)
        loss += curr_loss
        

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As will find it helpful to interleave yoa result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    W_h = np.copy(W)
    num_sample = 1
    for k in range(num_dim):    
        for j in range(num_classes):
            W_h[k, j] = W_h[k, j] + h
            scores = X.dot(W_h)
            margins = np.maximum(0, scores - scores[y] + 1)
            margins[y] = 0
            W_h[k, j] = W_h[k, j] - h
            loss_h = np.sum(margins) / num_train
            
            dW[k, j] += (loss_h - loss) / h
            dW[k, j] /= num_train
    '''
    for i in range(num_sample if num_train > num_sample else num_train):
        for k in range(num_dim):    
            for j in range(num_classes):
                W_h[k, j] = W_h[k, j] + h
                loss_h = L_i_vectorized(X[i], y[i], W_h)
                W_h[k, j] = W_h[k, j] - h
                dW[k, j] += (loss_h - loss) / h
    '''
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
    num_train = X.shape[0]
    scores = X.dot(W)
    margins = np.maximum(0, scores - scores[y] + 1)
    margins[y] = 0
    loss = np.sum(margins) / num_train
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
