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
    
    '''for i in range(num_train):
        curr_loss = L_i_vectorized(X[i], y[i], W)
        loss += curr_loss
        
    '''
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As will find it helpful to interleave yoa result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    '''
    num_dim = W.shape[0]
    h = 0.00001
    for d in range(W.shape[0]):
        for c in range(W.shape[1]):
            old_value = W[d, c]
            W[d, c] = old_value + h
            loss_h = L_i_vectorized(X, y, W)
            loss_h /= num_train
            loss_h += reg * np.sum(W * W)
            W[d, c] = old_value
            dW[d, c] = (loss - loss_h) / h
    '''
    
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train): 
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        num_classes_over_margin = 0
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                num_classes_over_margin += 1
        dW[:, y[i]] -= num_classes_over_margin * X[i]
    
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    dW /= num_train
    dW += reg * 2 * np.sum(W)
    
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
        
    scores = X @ W
    N = scores.shape[0]
    C = scores.shape[1]
    D = W.shape[0]
    
    corresponding_values = np.choose(y, scores.T)[:,None]
    rest_of_values = scores[np.arange(C) != y[:,None]].reshape(N, C -1)
    margin = rest_of_values - corresponding_values + 1
    rectified = np.where(margin > 0, margin, 0)
    loss = np.sum(rectified)/N + reg * np.sum(W * W)
    
    '''
    corresponding_mask = np.full(X.shape, False)
    corresponding_mask[y] = True
    
    corresponding_scores = np.ma.masked_array(X, corresponding_mask)
    rest_of_scores = np.array(np.ma.masked_array(X, corresponding_mask).reshape(-1, 1))
    
    margin = rest_of_scores - corresponding_scores + 1 # 500 x 10
    loss = 1 / N * np.maximum(margin, 0).sum() + reg * np.sum(W)
    '''

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

    indicated = (rectified > 0).astype(int)
    indicated = np.insert(indicated, np.arange(N) * (C-1) + y, -indicated.sum(axis=1)).reshape((N, C))
    dW = X.T @ indicated / N + 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
