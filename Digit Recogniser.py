# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 18:14:31 2020

@author: Matthew Barber

Logistic regression, neural network and SVM models to read greyscale
digits.
"""

import numpy as np
from scipy import special, optimize

FILENAME = 'data.csv'
CVSIZE = 0.2
TESTSIZE = 0.2

LABELCOUNT = 10

LRREGULARISATIONCONSTS = (0, 0.3, 1, 3, 10)
NNREGULARISATIONCONSTS = (0, 0.3, 1, 3)
SVMREGULARISATIONCONSTS = (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30)
SIGMAS = (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30)

def main():
    (trainingData, cvData, testData) = loadData(FILENAME)
    logisticRegression(trainingData, cvData, testData)
    neuralNetwork(1, 25, trainingData, cvData, testData)
    supportVectorMachine(trainingData, cvData, testData)

def logisticRegression(trainingData, cvData, testData):
    """
    Input: trainingData, cvData, testData: arrays of floats.
    
    Returns: None.
    
    ASSUME: all data sets have the same number of columns and at least 2
    columns.
    
    Train regularised logistic regression and print accuracy on all 3
    data sets.
    """
    paramCount = LABELCOUNT * np.shape(trainingData)[1]
    initParams = np.zeros(paramCount,)
    params = learnParams(LRCostAndGrad, initParams, LRREGULARISATIONCONSTS,
                         trainingData, cvData, 50)
    predictions = predictLR(params, trainingData, cvData, testData)
    print("Logistic Regression:")
    accuracy(predictions, trainingData, cvData, testData)
def LRCostAndGrad(params, C, data):
    """
    Input: params, data: arrays of floats. C: float.
    
    Returns: tuple of float followed by array of floats.
    
    ASSUME: params has shape (LABELCOUNT*n,), where n is the number of
    columns in data. data has at least 2 columns. Last column of data
    consists of ints from 0 to LABELCOUNT-1 inclusive.
    
    Return cost w/ reg. const C & grad of cost. Expected output in
    data's last col.
    """
    theta = shapeParams(params, LABELCOUNT)
    (X, y) = splitInOut(data)
    m = np.shape(X)[0]
    cost = 0
    
    for i in range(LABELCOUNT):
        (thetaTemp, yTemp) = convertLabels(theta, y, i)
        toRegularise = LRRegularisationTerms(thetaTemp)
        hypothesis = binaryLRHypothesis(thetaTemp, X)
        
        cost += binaryLRCost(toRegularise, C, hypothesis, yTemp, m)
        
        try:
            gradMatrix = np.concatenate((gradMatrix,
                                         binaryLRGrad(toRegularise, C, X,
                                                      hypothesis, yTemp, m)))
        # On the first iteration, gradMatrix is undefined.
        except NameError:
            gradMatrix = binaryLRGrad(toRegularise, C, X, hypothesis, yTemp, m)
    
    return (cost, np.ravel(gradMatrix))
def binaryLRCost(toRegularise, C, hypothesis, y, m):
    """
    Input: toRegularise, hypothesis, y: arrays of floats. C: float. m:
    int.
    
    Returns: float.
    
    ASSUME: toRegularise, hypothesis and y are vectors. hypothesis and y
    have m elements. The elements of y are all 0s and 1s. m is positive.
    
    Produce the regularised cost w/ reg. const C. y is the expected
    output.
    """
    # Calculate unregularised cost
    cost = (np.vdot(y-1, np.log(1-hypothesis))
            - np.vdot(y, np.log(hypothesis)))
    # Add regularisation term
    cost += C * np.vdot(toRegularise, toRegularise) / 2
    
    return cost / m
def binaryLRGrad(toRegularise, C, X, hypothesis, y, m):
    """
    Input: toRegularise, X, hypothesis, y: arrays of floats. C: float.
    m: int.
    
    Returns: array of floats.
    
    ASSUME: toRegularise, hypothesis and y are column vectors. X,
    hypothesis and y have m elements. X has the same number of columns
    as toRegularise has elements. The elements of y are all 0s and 1s. m
    is positive.
    
    Produce row vec. grad of LR cost fun. w/ reg. const C. y is the
    expected output.
    """    
    return (np.matmul(X.T, hypothesis-y)+C*toRegularise).T / m
def totalLRHypothesis(params, X, m):
    """
    Input: params, X: arrays of floats. m: int.
    
    Returns: array of floats.
    
    ASSUME: params has shape (m*n,), where n is the number of columns in
    X. m is positive.
    
    Produce hypo. matrix, where (i, j) entry is prob. that jth row of X
    has label i.
    """
    theta = shapeParams(params, m)
    
    return binaryLRHypothesis(X.T, theta)
def binaryLRHypothesis(theta, X):
    """
    Input: theta, X: arrays of floats.
    
    Returns: array of floats.
    
    ASSUME: X has the same number of columns as theta has rows.
    
    Produce sigmoid (applied element-wise) of X*theta.
    """
    return special.expit(np.matmul(X, theta))
def predictLR(params, trainingData, cvData, testData):
    """
    Input: params, trainingData, cvData, testData: arrays of floats.
    
    Returns: tuple of arrays of ints.
    
    ASSUME: all data sets have the same number of columns and at least 2
    columns. params has shape (LABELCOUNT*n,), where n is the number of
    columns in any of the data sets.
    
    Produce classification probabilities for each input for each of the
    data sets.
    """
    # Each column corresponds to a particular input.
    return tuple(map(lambda data: totalLRHypothesis(params,
                                                    splitInOut(data)[0],
                                                    LABELCOUNT),
                     (trainingData, cvData, testData)))
    
def LRRegularisationTerms(theta):
    """
    Input: theta: array of floats.
    
    Returns: array of floats.
    
    ASSUME: theta does not have shape (n,)
    
    Produce a copy of theta with the first element set to 0.
    """
    toRegularise = np.copy(theta)
    toRegularise[0][0] = 0 # We don't regularise the bias term.
    
    return toRegularise
def convertLabels(theta, y, label):
    """
    Input: theta, y: arrays of floats. label: int.
    
    Returns: tuple of arrays of ints.
    
    ASSUME: theta has LABELCOUNT rows. y is a column vector. label is
    between 0 and LABELCOUNT-1 inclusive.
    
    Produce labelth row of theta and vec. w/ True elts for each y elt.
    eq. to label.
    """
    return (np.array([theta[label]]).T, y == label)

def neuralNetwork(layerCount, layerSize, trainingData, cvData, testData):
    """
    Input: layerCount, layerSize: ints. trainingData, cvData, testData:
    arrays of floats.
    
    Returns: None.
    
    ASSUME: layerCount and layerSize are non-negative. All data sets
    have the same number of columns and at least 2 columns.
    
    Train n.n. w/ layerCount hidden layers & layerSize nodes per hidden
    layer.
    """
    n = np.shape(trainingData)[1]
    initParams = initNNParams(layerCount, layerSize, n)
    params = learnParams(NNCostAndGrad, initParams, NNREGULARISATIONCONSTS,
                         trainingData, cvData, 150, layerCount, layerSize)
    predictions = predictNN(params, layerCount, layerSize, trainingData,
                            cvData, testData)
    print("Neural Network:")
    accuracy(predictions, trainingData, cvData, testData)
def initNNParams(layerCount, layerSize, n):
    """
    Input: layerCount, layerSize, n: ints.
    
    Returns: array of floats.
    
    ASSUME: all inputs are positive.
    
    Produce randomly initialised parameters for the neural network.
    """
    # Initialise weights connecting input layer to first hidden layer.
    params = initConnectingParams(n-1, layerSize)
    
    # Initialise weights between hidden layers.
    for i in range(layerCount-1):
        params = np.concatenate((params,
                                 initConnectingParams(layerSize, layerSize)))
    
    # Initialise weights between final hidden layer and output layer.
    return np.concatenate((params,
                           initConnectingParams(layerSize, LABELCOUNT)))
def initConnectingParams(n1, n2):
    """
    Input: n1, n2: ints.
    
    Returns: array of ints.
    
    ASSUME: n1 and n2 are positive.
    
    Return ((n1+1) * n2,) array of values from Unif([-a, a)). a
    = sqrt(6/(n1+n2)).
    """
    a = np.sqrt(6/(n1+n2))
    
    return a * (2*np.random.rand((n1+1) * n2)-1)
def NNCostAndGrad(params, C, data, layerCount, layerSize):
    """
    Input: params, data: arrays of floats. layerCount, layerSize: ints.
    C: float.
    
    Returns: tuple of float followed by array of floats.
    
    ASSUME: params has shape (layerSize*n
                              + (layerCount-1)*layerSize*(layerSize+1)
                              + (layerSize+1)*LABELCOUNT,), where n is
    the number of columns in data. data has at least 2 columns. Last
    column of data consists of ints from 0 to LABELCOUNT-1 inclusive.
    layerCount and layerSize are positive.
    
    Return cost w/ reg. const C & grad of cost. Expected output is
    data's last col.
    """
    (X, y) = splitInOut(data)
    activations = forwardPropogation(params, layerCount, layerSize, X)
    
    cost = NNCost(params, C, activations, y)
    grad = NNGrad(params, C, activations, y)
    
    return (cost, grad)
def forwardPropogation(params, layerCount, layerSize, X):
    """
    Input: params, X: arrays of floats. layerCount, layerSize: ints.
    
    Returns: list of arrays of floats.
    
    ASSUME: params has shape (layerSize*n
                              + (layerCount-1)*layerSize*(layerSize+1)
                              + (layerSize+1)*LABELCOUNT,), where n is
    the number of columns in X. X has at least 2 columns. layerCount and
    layerSize are positive.
    
    Return activations of nodes in each layer. In a layer's array, 1 row
    per input.
    """
    # We do not include the bias terms in the final output.
    unbiasedX = X[:, 1:]
    if layerCount == 0:
        return [unbiasedX, totalLRHypothesis(params, X, LABELCOUNT).T]
    
    (m, n) = np.shape(X)
    firstLayerParamCount = layerSize*n
    # Compute the activations of the first hidden layer.
    biasExcludedActivation = totalLRHypothesis(params[:firstLayerParamCount],
                                               X, layerSize).T
    # We must include the bias terms in the input to the recursive call.
    biasIncludedActivation = addBias(biasExcludedActivation)
    
    # Propogate forward using the computed activations as the input
    # layer.
    nonInputActivations = forwardPropogation(params[firstLayerParamCount:],
                                             layerCount-1, layerSize,
                                             biasIncludedActivation)
    
    return [unbiasedX]+nonInputActivations
def NNCost(params, C, activations, y):
    """
    Input: params, y: arrays of floats. C: float. activations: list of
    arrays of floats.
    
    Returns: array of floats.
    
    ASSUME: params has shape (layerSize*n
                              + (layerCount-1)*layerSize*(layerSize+1)
                              + (layerSize+1)*LABELCOUNT,), for some
    layerCount, layerSize and n. layerCount and layerSize are positive
    and n > 1. activations has layerCount+2 entries. All arrays in
    activations have m rows, where m is the number of elements in y. The
    final array has LABELCOUNT columns and the others, excluding the
    first, have layerSize columns. y consists of ints from 0 to
    LABELCOUNT-1 inclusive.
    
    Produce n.n. cost w/ regularisation constant C. Expected output of
    network is y.
    """
    # Once we have propogated the activations forward to the last hidden
    # layer, we have a logistic regression model where the inputs are
    # the activations of the final hidden layer.
    m = np.shape(y)[0]
    (layerCount, layerSize, n) = NNArchitecture(activations)
    finalLayerParamCount = (layerSize+1) * LABELCOUNT
    finalLayerParams = params[-finalLayerParamCount:]
    LRData = np.concatenate((activations[-2], y), 1)
    unregularisedCost = LRCostAndGrad(finalLayerParams, 0, LRData)[0]
    cost = (unregularisedCost
            + NNRegularisationCost(params, C, layerCount, layerSize, n)/(2*m))
    
    return cost
def NNRegularisationCost(params, C, layerCount, layerSize, n):
    """
    Input: params: array of floats. C: float. layerCount, layerSize, n:
    ints.
    
    Returns: float.
    
    ASSUME: params has shape (layerSize*n
                              + (layerCount-1)*layerSize*(layerSize+1)
                              + (layerSize+1)*LABELCOUNT,), layerCount
    and layerSize are positive. n > 1.
    
    Produce C times the sum of the squares of weights not connected to a
    bias node.
    """
    cost = 0
    
    # Add regularisation of weights between input and first hidden
    # layer.
    layerIndex = layerSize*n
    theta = NNRegularisationTerms(params[:layerIndex], layerSize)
    cost += np.sum(np.square(theta))
    
    # Add regularisation of weights between hidden layers.
    weightsPerLayer = layerSize * (layerSize+1)
    for i in range(layerCount-1):
        nextLayerIndex = layerIndex+weightsPerLayer
        theta = NNRegularisationTerms(params[layerIndex:nextLayerIndex],
                                      layerSize)
        cost += np.sum(np.square(theta))
        layerIndex = nextLayerIndex
    
    # Add regularisation of weights between final hidden layer and
    # output layer.
    theta = NNRegularisationTerms(params[layerIndex:], LABELCOUNT)
    return C * (cost+np.sum(np.square(theta)))
def NNGrad(params, C, activations, y):
    """
    Input: params, y: arrays of floats. C: float. activations: list of
    arrays of floats.
    
    Returns: array of floats.
    
    ASSUME: params has shape (layerSize*n
                              + (layerCount-1)*layerSize*(layerSize+1)
                              + (layerSize+1)*LABELCOUNT,), for some
    layerCount, layerSize and n. layerCount and layerSize are positive
    and n > 1. activations has layerCount+2 entries. All arrays in
    activations have m rows, where m is the number of elements in y. The
    final array has LABELCOUNT columns and the others, excluding the
    first, have layerSize columns. y consists of ints from 0 to
    LABELCOUNT-1 inclusive.
    
    Produce grad of cost w.r.t. params with reg. const C. Expected
    output is y.
    """
    (layerCount, layerSize, n) = NNArchitecture(activations)
    firstLayerParamCount = layerSize*n
    deltas = backPropogation(params[firstLayerParamCount:], activations, y)
    
    return accumulateGrads(params, C, activations, deltas)
def backPropogation(params, activations, y):
    """
    Input: params, y: arrays of floats. activations: list of arrays of
    floats.
    
    Returns: list of arrays of floats.
    
    ASSUME: params has shape (layerSize*n
                              + (layerCount-1)*layerSize*(layerSize+1)
                              + (layerSize+1)*LABELCOUNT,), for some
    layerCount, layerSize and n. layerCount and layerSize are positive
    and n > 1. activations has layerCount+2 entries. All arrays in
    activations have m rows, where m is the number of elements in y. The
    final array has LABELCOUNT columns and the others, excluding the
    first, have layerSize columns. y consists of ints from 0 to
    LABELCOUNT-1 inclusive.
    
    Return grad of cost w.r.t. pre-sigmoid activations of nodes in
    non-input layers.
    """
    (layerCount, layerSize, n) = NNArchitecture(activations)
    weightsPerLayer = layerSize * (layerSize+1) # between hidden layers
    deltas = [activations[-1]-expectedActivations(y)]
    prevLayerIndex = len(params)
    
    for i in range(layerCount):
        layerIndex = (layerCount-i-1) * weightsPerLayer
        tempParams = params[layerIndex:prevLayerIndex]
        prevLayerIndex = layerIndex
        theta = shapeParams(tempParams, nodeCount(i, layerSize))
        tempActivations = activations[layerCount-i]
        sigmoidGrad = np.multiply(tempActivations, 1-tempActivations)
        newDelta = np.multiply(np.matmul(deltas[0], theta[:, 1:]), sigmoidGrad)
        deltas = [newDelta] + deltas
    
    return deltas
def accumulateGrads(params, C, activations, deltas):
    """
    Input: params: array of floats. C: float. activations, deltas: lists
    of arrays of floats.
    
    Returns: array of floats.
    
    ASSUME: params has shape (layerSize*n
                              + (layerCount-1)*layerSize*(layerSize+1)
                              + (layerSize+1)*LABELCOUNT,), where n-1 is
    the number of columns in activations[0]. activations has
    layerCount+2 entries. All arrays in activations have m rows, for
    some m. The final array in activations has LABELCOUNT columns and
    the others, excluding the first, have layerSize columns.
    
    Produce grad of n.n. cost w/ reg. const C. deltas stores
    backpropagated errors.
    """
    (layerCount, layerSize, n) = NNArchitecture(activations)
    weightsPerLayer = layerSize * (layerSize+1) # between hidden layers
    
    # Compute grad w.r.t. params connecting input to first hidden layer.
    layerIndex = layerSize * n
    grad = np.array([])
    grad = interLayerGrad(params[:layerIndex], C, activations[0], deltas[0],
                          grad)
    
    # Compute grad w.r.t. params connecting hidden layers.
    for i in range(1, layerCount):
        nextLayerIndex = layerIndex+weightsPerLayer
        grad = interLayerGrad(params[layerIndex:nextLayerIndex], C,
                              activations[i], deltas[i], grad)
        layerIndex = nextLayerIndex
    
    # Compute grad w.r.t. params connecting final hidden layer to output
    # layer.
    return interLayerGrad(params[layerIndex:], C, activations[-2], deltas[-1],
                          grad)
def interLayerGrad(params, C, activation, delta, previousGrad):
    """
    Input: params, activation, delta, previousGrad: array of floats. C:
    float.
    
    Returns: array of floats.
    
    ASSUME: params has shape ((n1+1) * n2,), where activation has shape
    (m, n1) and delta has shape (m, n2) for some positive m.
    previousGrad has shape (n,) for some non-negative n.
    
    Append grad of cost w.r.t. params joining 2 consecutive layers to
    previousGrad.
    """
    m = np.shape(activation)[0]
    activation = addBias(activation)
    unregularisedGrad = np.matmul(delta.T, activation)
    regularisationGrad = C * NNRegularisationTerms(params, np.shape(delta)[1])
    totalGrad = (unregularisedGrad+regularisationGrad) / m
    
    return np.concatenate((previousGrad, np.ravel(totalGrad)))
def NNArchitecture(activations):
    """
    Input: activations: list of arrays of ints.
    
    Returns: tuple of ints.
    
    ASSUME: activations has at least 3 elements. The first, the second
    and the last do not have shape (n,).
    
    Produce no. of layers, no. of nodes per hidden layer, no. of input features + 1.
    """
    return (len(activations)-2, np.shape(activations[1])[1],
            np.shape(activations[0])[1]+1)
def NNRegularisationTerms(params, m):
    """
    Input: params: array of floats. m: int.
    
    Returns: array of floats.
    
    ASSUME: params has shape (n,) for some n divisible by m. m is positive.
    
    Shape params into a (m, n/m) shaped array and set first column to 0s. 
    """
    theta = shapeParams(params, m)
    theta[:, 0] = 0
    return theta
def nodeCount(i, layerSize):
    """
    Input: i: int.
    
    Returns: int.
    
    ASSUME: layerSize is an int.
    
    Produce LABELCOUNT if i is 0, layerSize otherwise.
    """
    if i == 0:
        return LABELCOUNT
    
    return layerSize
def expectedActivations(y):
    """
    Input: y: array of floats.
    
    Returns: array of floats.
    
    ASSUME: y is a column vector whose elements are ints from 0 to
    LABELCOUNT-1 inclusive.
    
    Return array w/ a 1 in a single col. per row. Represents y value for
    each input.
    """
    m = np.shape(y)[0]
    Y = np.zeros((m, LABELCOUNT))
    for i in range(m):
        Y[i, int(y[i])] = 1
    
    return Y
def predictNN(params, layerCount, layerSize, trainingData, cvData, testData):
    """
    Input: params, trainingData, cvData, testData: arrays of floats.
    
    Returns: tuple of arrays of floats.
    
    ASSUME: layerCount and layerSize are positive. All data sets have
    the same number of columns and at least 2 columns. params has shape
    (layerSize*n + (layerCount-1)*layerSize*(layerSize+1)
     + (layerSize+1)*LABELCOUNT,), where n is the number of columns in
    any of the data sets.
    
    Produce classification probabilities for each input for each of the
    data sets.
    """
    return tuple(map(lambda data:
                     forwardPropogation(params, layerCount, layerSize,
                                        splitInOut(data)[0])[-1].T,
                     (trainingData, cvData, testData)))

def supportVectorMachine(trainingData, cvData, testData):
    """
    Input: trainingData, cvData, testData: arrays of floats.
    
    Returns: None.
    
    ASSUME: all data sets have the same number of columns and at least 2
    columns.
    
    Train regularised SVM with Gaussian kernel and print accuracy on the
    data sets.
    
    !!!
    """
    return None

def learnParams(costFun, initParams, regularisationConsts, trainingData,
                cvData, maxIter, *args):
    """
    Input: costFun: function. initParams, trainingData, cvData: arrays
    of floats. regularisationConsts: tuple of floats. maxIter: int.
    
    Returns: array of floats.
    
    ASSUME: initParams, C, data, *args are valid inputs to costFun where
    C may be any float and data may be trainingData or cvData.
    cvData. costFun returns a tuple containing a float and the gradient
    of that float with respect to costFun's first argument. maxIter is
    non-negative.
    
    Find params to minimise cost on trainingData. Pick reg. const with
    cvData.
    """
    for C in regularisationConsts:
        params = train(costFun, initParams, C, trainingData, maxIter, *args)
        cost = costFun(params, 0, cvData, *args)[0]
        
        try:
            if cost < bestCost:
                (bestParams, bestCost) = (params, cost)
        # On the first iteration, bestCost is undefined.
        except NameError:
            (bestParams, bestCost) = (params, cost)
        
    return bestParams
def train(costFun, initParams, C, trainingData, maxIter, *args):
    """
    Input: costFun: function. initParams, trainingData: arrays of
    floats. C: float. maxIter: int.
    
    Returns: array of floats.
    
    ASSUME: initParams, C, trainingData, *args are valid inputs to
    costFun where C may be any float. costFun returns a tuple containing
    a float and the gradient of that float with respect to costFun's
    first argument. maxIter is non-negative.
    
    Produce params to minimise cost on trainingData w/ reg. const C.
    """
    return optimize.minimize(costFun, initParams, (C, trainingData, *args),
                             method='CG', jac=True,
                             options={'maxiter':maxIter}).x

def accuracy(hypotheses, trainingData, cvData, testData):
    """
    Input: hypotheses: tuple of arrays of floats. trainingData, cvData,
    testData: arrays of floats.
    
    Returns: None.
    
    ASSUME: hypotheses has three elements, all of which have LABELCOUNT
    rows. hypotheses[0], hypotheses[1] and hypotheses[2] have the same
    number of columns as trainingData, cvData and testData have rows
    respectively. All data sets have the same number of columns and at
    least 2 columns.
    
    Print the classification accuracy of the predictions on each of the
    data sets.
    """
    ySets = tuple(map(lambda data: splitInOut(data)[1],
                      (trainingData, cvData, testData)))
    setNames = ('Training', 'Cross-Validation', 'Test')
    predictions = tuple(map(lambda hypothesis:
                            np.array([np.argmax(hypothesis, 0)]).T,
                            hypotheses))
    
    for i in range(3):
        setAccuracy(setNames[i], predictions[i], ySets[i])
    print("\n")
    
def setAccuracy(setName, prediction, y):
    """
    Input: setName: str. prediction: array of ints. y: array of floats.
    
    Returns: None.
    
    ASSUME: prediction and y have the same dimensions.
    
    Print the proportion of elts. of prediction eq. to their
    corresponding y elt.
    """
    print(setName, "Accuracy:", np.mean(prediction == y))

def shapeParams(params, m):
    """
    Input: params: array of floats. m: int.
    
    Returns: array of floats.
    
    ASSUME: params has shape (n,) for some n divisible by m. m is
    positive.
    
    Shape params into a matrix with m rows.
    """
    return np.reshape(params, (m, int(len(params)/m)))

def loadData(fileName):
    """
    Input: fileName: str.
    
    Returns: Tuple of array of floats.
    
    ASSUME: fileName is the path to a valid csv file.
    
    Create training, cross-validation and test data sets from the file.
    """
    data = np.loadtxt(fileName, delimiter = ",")
    # We shuffle the data to try to ensure that the training data and
    # the test data have a similar composition.
    np.random.shuffle(data)
    
    m = np.shape(data)[0]
    cvIndex = int(np.floor(m*CVSIZE))
    testIndex = cvIndex + int(np.floor(m*TESTSIZE))
    (cvData, testData, trainingData) = splitSets(data, cvIndex, testIndex)
    
    return (trainingData, cvData, testData)
def splitInOut(data):
    """
    Input: data: array of floats.
    
    Returns: tuple of arrays of floats.
    
    ASSUME: data has at least 2 columns.
    
    Produce output (final col. of data) and input with col. of 1s added
    to the left.
    """
    m = np.shape(data)[0]
    
    return (addBias(data[:, :-1]), np.reshape(data[:, -1], (m, 1)))
def splitSets(data, firstIndex, secondIndex):
    """
    Input: data: array of floats. firstIndex, secondIndex: ints.
    
    Returns: tuple of arrays of floats.
    
    ASSUME: 0 <= firstIndex <= secondIndex <= m, where m is the number
    of rows in data.
    
    Split data at the given row indices into three arrays.
    """
    return (data[:firstIndex], data[firstIndex:secondIndex],
            data[secondIndex:])
def addBias(X):
    """
    Input: X: array of floats.
    
    Return: array of floats.
    
    Add column of 1s to left of X.
    """
    m = np.shape(X)[0]
    return np.concatenate((np.ones((m, 1)), X), 1)

if __name__ == '__main__':
    main()