from preprocess_image import read_images_from_folder
from preprocess_image import _preprocess
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import random
from sklearn.metrics import f1_score

from Dense import Dense
from Activation import Activation
from Loss import compute_BCE_loss
from Predict import predict
from preprocess_image import preprocess_image

output = {}
seed = 1
np.random.seed(seed)

class Conv():
    def __init__(self, filter_size=2, input_channel=3, output_channel=8, pad=1, stride=1, seed=1):

        self.filter_size= filter_size
        self.input_channel=input_channel
        self.output_channel=output_channel
        self.seed = seed
        self.pad = pad
        self.stride = stride

        self.parameters = {'W':None, 'b': None}
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        self.parameters -- python dictionary containing your parameters:
        W -- weight matrix of shape (filter_size, filter_size, input channel size, output channel size)
        b -- bias vector of shape (1, 1, 1, output channel size)
        """
        np.random.seed(seed)
        sd = np.sqrt(6.0 / (self.input_channel + self.output_channel))
        W = np.random.uniform(-sd, sd, (self.filter_size,self.filter_size,self.input_channel,self.output_channel))
        b = np.zeros((1, 1, 1, self.output_channel))

        assert(W.shape == (self.filter_size,self.filter_size,self.input_channel,self.output_channel))
        assert(b.shape == (1,1,1,self.output_channel))

        self.parameters['W'] = W
        self.parameters['b'] = b
        
def zero_pad(X, pad):
    """
    Pad all images in the dataset X with zeros. The padding should be applied to both the height and width of each image.

    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C), where m represent the number of examples.
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """

    ### START CODE HERE ###
    X_pad = np.pad(X, ((0,0),(pad, pad), (pad, pad), (0,0)), "constant", constant_values=(0,0))
    ### END CODE HERE ###

    return X_pad

def conv_single_step(self, a_slice_prev, W, b):
        """
        Arguments:
        a_slice_prev -- slice of previous activation layer output with shape (filter_size, filter_size, n_C_prev)
        W -- Weight parameters contained in a window - matrix of shape (filter_size, filter_size, n_C_prev)
        b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

        Returns:
        Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
        """

        ### START CODE HERE ### (≈ 3 lines)
        # Step 1: Element-wise product to a_slice_prev and W
        filtered = a_slice_prev * W
        # Step 2: Sum all values to get a single scalar
        Z = np.sum(filtered)
        # Step 3: Add the bias
        Z = Z + np.squeeze(b)
        ### END CODE HERE ###

        return Z


def forward(self, A_prev):
    """
    Implements the forward propagation for a convolution layer

    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)

    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    """

    ### START CODE HERE ###
    # Retrieve dimensions from A_prev's shape
    print(A_prev.shape)
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = self.parameters["W"].shape


    # Step 1: Output Dimension Calculation
    pad = self.pad
    stride = self.stride
    n_H = int(np.floor(n_H_prev - self.filter_size + 2 * pad)/self.stride + 1)
    n_W = int(np.floor(n_W_prev - self.filter_size + 2 * pad)/self.stride + 1)

    # Initialize the output volume Z with zeros
    Z = np.zeros((m, n_H, n_W, n_C))
    # Step 2: Padding
    A_prev_pad = zero_pad(A_prev, pad)

    # Step 3: Loop Through Training Examples
    for i in range(m):                                 # loop over the batch of training examples
        for h in range(n_H):                           # loop over vertical axis of the output volume
            for w in range(n_W):                       # loop over horizontal axis of the output volume
                for c in range(n_C):                   # loop over channels (= #filter) of the output volume


                    # Step 3-1: Extracting slices
                    vert_start = h * self.stride
                    vert_end = vert_start + self.filter_size
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + self.filter_size
                    a_slice_prev = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]

                    # Step 3-2: Applying Filters
                    Z[i, h, w, c] = self.conv_single_step(a_slice_prev, self.parameters["W"][:,:,:,c], self.parameters["b"][:,:,:,c])

    ### END CODE HERE ###

    # Making sure your output shape is correct
    assert(Z.shape == (m, n_H, n_W, n_C))

    # Save information in "cache" for the backward pass
    self.cache = A_prev

    return Z

Conv.forward = forward

Conv.conv_single_step = conv_single_step

def backward(self, dZ):
    """
    Implement the backward propagation for a convolution layer

    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)

    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
                numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    """


    A_prev = self.cache

    ### START CODE HERE ###

    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = self.parameters["W"].shape

    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape

    # Step 1: Initialize Gradients
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # Step 2: Padding
    A_prev_pad = zero_pad(A_prev, self.pad)
    dA_prev_pad = zero_pad(dA_prev, self.pad)

    # Step 3: Loop Through Training Examples
    for i in range(m):                         # loop over the batch of training examples
        for h in range(n_H):                   # loop over vertical axis of the output volume
            for w in range(n_W):               # loop over horizontal axis of the output volume
                for c in range(n_C):           # loop over the channels of the output volume

                    # Step 3-1: Extracting slices
                    vert_start = h * self.stride
                    vert_end = vert_start + self.filter_size
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + self.filter_size
                    a_slice = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]

                    # Step 3-2: Update the Gradients
                    dA_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :] += self.parameters["W"][:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] = dW[:,:,:,c] + (1 / m)* a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] = db[:,:,:,c] + (1/m) * dZ[i, h, w, c]

        # Step 4: Remove Padding
        dA_prev[i, :, :, :] = dA_prev_pad[i, self.pad:-self.pad, self.pad:-self.pad, :]

    ### END CODE HERE ###

    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    self.dW = dW
    self.db = db

    return dA_prev

Conv.backward = backward

def update(self, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    learning rate -- step size
    """

    ### START CODE HERE ###
    self.parameters["W"] = self.parameters["W"] - learning_rate * self.dW
    self.parameters["b"] = self.parameters["b"] - learning_rate * self.db
    ### END CODE HERE ###

Conv.update = update
class MaxPool():
    def __init__(self, pool_size=2, stride=2):

        self.pool_size = pool_size
        self.stride = stride

    def create_mask_from_window(self, x):
        """
        Creates a mask from an input x to identify the max entry of x.

        Arguments:
        x -- Array of shape (filter_size, filter_size)

        Returns:
        mask -- Array of the same shape as filter, contains a True at the position corresponding to the max entry of x.
        """

        mask = x == np.max(x)

        return mask
    def forward(self, A_prev):
        """
        Implements the forward pass of the max pooling layer

        Arguments:
        A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)

        Returns:
        A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
        """

        ### START CODE HERE ###
        # retrieve dimensions from the input shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape


        # Step 1: Output Dimension Calculation
        n_H = int((n_H_prev - self.pool_size)/self.stride) + 1
        n_W = int((n_W_prev - self.pool_size)/self.stride) + 1
        n_C = n_C_prev

        # initialize output matrix A with zeros
        A = np.zeros((m, n_H, n_W, n_C))

        # Step 2: Loop Through Training Examples
        for i in range(m):                           # loop over the batch of training examples
            for h in range(n_H):                     # loop on the vertical axis of the output volume
                for w in range(n_W):                 # loop on the horizontal axis of the output volume
                    for c in range (n_C):            # loop over the channels of the output volume

                        # Step 2-1: Extracting slices
                        vert_start = h * self.stride
                        vert_end = vert_start + self.pool_size
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.pool_size
                        a_prev_slice = A_prev[i][vert_start:vert_end, horiz_start:horiz_end, c]

                        # Step 2-2: Applying Maxpooling
                        A[i, h, w, c] = np.max(a_prev_slice)

        ### END CODE HERE ###

        # Store the input in "cache" for backward pass
        self.cache = A_prev

        # Making sure your output shape is correct
        assert(A.shape == (m, n_H, n_W, n_C))

        return A

    def backward(self, dA):
        """
        Implements the backward pass of the max pooling layer

        Arguments:
        dA -- gradient of cost with respect to the output of the pooling layer, same shape as A

        Returns:
        dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
        """

        # Retrieve information from cache
        A_prev = self.cache

        ### START CODE HERE ###

        # Retrieve dimensions from A_prev's shape and dA's shape
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        m, n_H, n_W, n_C = dA.shape

        # Step 1: Initialize Gradients
        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))

        # Step 2: Loop Through Training Examples
        for i in range(m):                           # loop over the batch of training examples
            for h in range(n_H):                     # loop on the vertical axis of the output volume
                for w in range(n_W):                 # loop on the horizontal axis of the output volume
                    for c in range (n_C):            # loop over the channels of the output volume

                        # Step 2-1: Extracting slices
                        vert_start = h * self.stride
                        vert_end = vert_start + self.pool_size
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.pool_size
                        a_prev_slice = A_prev[i][vert_start:vert_end, horiz_start:horiz_end, c]

                        # Step 2-2: Pass through the Gradients
                        mask = self.create_mask_from_window(a_prev_slice)
                        dA_prev[i][vert_start:vert_end, horiz_start:horiz_end, c] = mask * dA[i, h, w, c]

        ### END CODE HERE ###

        # Make sure your output shape is correct

        assert(dA_prev.shape == A_prev.shape)

        return dA_prev

class Flatten():
    def __init__(self):
        pass

    def forward(self, A_prev):
        """
        Implements the forward pass of the flatten layer

        Arguments:
        A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)

        Returns:
        A -- output of the flatten layer, a 2-dimensional array of shape (m, (n_H_prev * n_W_prev * n_C_prev))
        """

        # Save information in "cache" for the backward pass
        self.cache = A_prev.shape

        ### START CODE HERE ###
        A = A_prev.reshape(A_prev.shape[0], -1)
        ### END CODE HERE ###
        return A

    def backward(self, dA):
        """
        Implements the backward pass of the flatten layer

        Arguments:
        dA -- Input data, a 2-dimensional array

        Returns:
        dA_prev -- An array with its original shape (the output shape of its' previous layer).
        """
        ### START CODE HERE ###
        dA_prev = dA.reshape(self.cache)
        ### END CODE HERE ###
        return dA_prev

class Model():
    def __init__(self):
        self.layers=[]

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        A = X

        ### START CODE HERE ###
        for l in range(len(self.layers)):
            A = self.layers[l].forward(A)
        ### END CODE HERE ###
        return A

    def backward(self, AL=None, Y=None):
        L = len(self.layers)

        ### START CODE HERE ###
        e = 10**(-5)
        dAL = (-1)*(Y/(AL + e) - (1 - Y)/(1 - AL + e))
        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL". Outputs: "dA_prev"
        dZ = self.layers[L-1].backward(dA=dAL)
        dA_prev = self.layers[L-2].backward(dZ)
        # Loop from l=L-2 to l=0
        for l in reversed(range(L-2)):
            dA_prev = self.layers[l].backward(dA_prev)
        ### END CODE HERE ###

        return dA_prev

    def update(self, learning_rate):

        # Only convolution layer and dense layer have to update parameters
        ### START CODE HERE ###
        for l in range(len(self.layers)):
            if self.layers[l].__class__.__name__ == "Conv" or self.layers[l].__class__.__name__ == "Dense":
                self.layers[l].update(learning_rate)
        ### END CODE HERE ###


model_data = np.load('CNN_model.npy', allow_pickle=True)
print(model_data)
model=Model()
model.add(Conv(filter_size=3, input_channel=1, output_channel=8, pad=1, stride=2))
model.add(MaxPool(pool_size=2, stride=2))
model.add(Flatten())
model.add(Dense(16200, 6400))
model.add(Activation("relu", None))
model.add(Dense(6400, 2048))
model.add(Activation("relu", None))
model.add(Dense(2048, 1))
model.add(Activation("sigmoid", None))

num = 0
for i in range(len(model.layers)):
    if model.layers[i].__class__.__name__ == "Conv" or model.layers[i].__class__.__name__ == "Dense":
        model.layers[i].parameters = model_data[num]
        num += 1


preprocess_test = np.array(read_images_from_folder("D:\MachineLearning_FinalProject\code\\fake_mix_photo"), dtype="object")
for i in range(len(preprocess_test)):
    preprocess_test[i] = _preprocess(preprocess_test[i])
print(preprocess_test.shape)

preprocess_test = preprocess_test.reshape(preprocess_test.shape[0], preprocess_test.shape[1], preprocess_test.shape[2],1)


test_labels = np.zeros((preprocess_test.shape[0], 1))
label1 = [0,1,2,3,4,5,6,7,11,12,13,14,15,16,17,18,21,26,27,28,29,30,31,32,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52]
label_bin = [8,9,10,61,63,67,69,70,72,73,74,75,77,82,84,85,86,87,97,98,99,100,101,102,104,105,106,107,108,109,110,111]
label_black = [53,54,55,56,57,58,59,60,62,64,65,66,68,71,76,78,79,80,81,83,88,89,90,91,92,93,94,95,96,103]
label_non = [19,20,22,23,24,25,33,34,35,36]
error_bin = 0
error_black = 0
error_non = 0
for i in label1:
    test_labels[i] = 1


predict_testing = predict(model, preprocess_test, test_labels)

for i in range(len(predict_testing)):
    print(f'number{i}, predict: {predict_testing[i]}, truth: {test_labels[i]}')
    if i in label_bin and predict_testing[i] == 1:
        error_bin = error_bin + 1
    if i in label_black and predict_testing[i] == 1:
        error_black = error_black + 1
    if i in label_non and predict_testing[i] == 1:
        error_non = error_non + 1
