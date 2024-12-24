#4720 + 2606 = 7326
# import model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from PIL import Image
import random
#from sklearn.metrics import f1_score
#from matplotlib.animation import FuncAnimation
#from google.colab import drive
#drive.mount('/content/drive')
from sklearn.metrics import f1_score

def _preprocess(image_arr, th=0.4):
        image_arr[image_arr > th] = 1.0
        image_arr[image_arr <= th] = 0.0
        return image_arr

from PIL import Image

def image_resize(path):
    # path : the image path you want to resize
    # img_resized: img have resize to 360*360
    #open picture
    img = Image.open(path)
    # resize to 360*360
    img_resized = img.resize((180, 180), Image.LANCZOS)
    return img_resized
def read_images_from_folder(folder_path):
    # read images from 'folder_path'
    # flat the image and push into 'images' array
    print(f'reading from {folder_path}...')
    images = []
    length = len(os.listdir(folder_path))
    i = 0
    for filename in os.listdir(folder_path):
        i = i + 1
        progress = ((i + 1)/length)*100
        if i % 500 == 0:
            print(f"progress: {progress} %")
        if filename.lower().endswith('.png') or filename.lower().endswith('.jpg'):
            file_path = os.path.join(folder_path, filename)
            try:
                # 打開圖片
                with Image.open(file_path) as img:
                    # 將圖片轉為灰階
                    img = img.convert('L')
                    # 檢查圖片尺寸是否為 360x360
                    if img.size != (180, 180):
                        img = image_resize(file_path)
                        img.save(file_path)
                        print(file_path)
                        print(img.size)
                    if img.size == (180, 180):
                        # 將圖片轉換為 np.array 並進行標準化 (值範圍從 0 到 1)
                        img_array = np.array(img) / 255.0
                        # 將圖片展平 (flatten)
                        img_array = img_array.flatten()
                        #print(img_array)
                        images.append(img_array)
                        #print(images[0])
                        #print(f'image size: {len(images[0])}')
                    else:
                        print(f"圖片尺寸為 {img.size}, 不是 180x180: {filename}")
            except UnidentifiedImageError:
                print(f"無法識別圖片文件: {filename}")
            except Exception as e:
                print(f"無法讀取圖片: {filename}, 錯誤: {e}")
    print(f"finish reading from {folder_path}")
    return images

class Dense():
    def __init__(self, n_x, n_y, seed=1):
        self.n_x = n_x
        self.n_y = n_y
        self.seed = seed
        self.initialize_parameters()

    def initialize_parameters(self):

        sd = np.sqrt(6.0 / (self.n_x + self.n_y))
        np.random.seed(self.seed)
        W = np.random.uniform(-sd, sd, (self.n_y, self.n_x)).T      # the transpose here is just for the code to be compatible with the old codes
        b = np.zeros((1, self.n_y))

        assert(W.shape == (self.n_x, self.n_y))
        assert(b.shape == (1, self.n_y))

        self.parameters = {"W": W, "b": b}

    def forward(self, A):
        # GRADED FUNCTION: linear_forward
        ### START CODE HERE ###
        W = self.parameters["W"]
        b = self.parameters["b"]
        Z = np.dot(A, W) + b # matrix multiplication and addition
        self.cache = (A, W, b)

        ### END CODE HERE ###

        assert(Z.shape == (A.shape[0], self.parameters["W"].shape[1]))

        return Z

    def backward(self, dZ):

        A_prev, W, b = self.cache
        m = A_prev.shape[0]

        # GRADED FUNCTION: linear_backward
        ### START CODE HERE ###
        self.dW = (1 / m) * np.dot(A_prev.T, dZ) #gradient of loss with respect to weights
        self.db = (1 / m) * np.sum(dZ, axis=0, keepdims=True) #gradient of loss with respect to bias
        dA_prev = np.dot(dZ, W.T)
        ### END CODE HERE ###

        assert (dA_prev.shape == A_prev.shape)
        assert (self.dW.shape == self.parameters["W"].shape)
        assert (self.db.shape == self.parameters["b"].shape)

        return dA_prev

    def update(self, learning_rate):
    # GRADED FUNCTION: linear_update_parameters
    ### START CODE HERE ###
      self.parameters["W"] = self.parameters["W"] - learning_rate * self.dW
      self.parameters["b"] = self.parameters["b"] - learning_rate * self.db
      ### END CODE HERE ###
class Activation():
    def __init__(self, activation_function, loss_function):
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.cache = None

    def forward(self, Z):
        if self.activation_function == "sigmoid":
            # GRADED FUNCTION: sigmoid_forward
            ### START CODE HERE ###
            Z = np.array(Z, dtype=float)

            A = 1 / (1 + np.exp(-Z))
            self.cache = Z
            ### END CODE HERE ###

            return A
        elif self.activation_function == "relu":
            # GRADED FUNCTION: relu_forward
            ### START CODE HERE ###
            A = np.maximum(0, Z)
            self.cache = Z
            ### END CODE HERE ###

            assert(A.shape == Z.shape)

            return A
        elif self.activation_function == "softmax":
            # GRADED FUNCTION: softmax_forward
            ### START CODE HERE ###
            A = np.exp(Z - np.max(Z, axis=1, keepdims=True)) / np.sum(np.exp(Z - np.max(Z, axis=1, keepdims=True)), axis=1, keepdims=True)
            self.cache = Z
            ### END CODE HERE ###

            return A
        elif self.activation_function == "linear":
            self.cache = Z.copy()
            return Z

        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function}")


    def backward(self, dA=None, Y=None):
        if self.activation_function == "sigmoid":
            # GRADED FUNCTION: sigmoid_backward
            ### START CODE HERE ###
            Z = np.array(self.cache)
            A = 1 / (1 + np.exp(-Z))
            dZ = dA * A * (1 - A)
            ### END CODE HERE ###

            assert (dZ.shape == Z.shape)

            return dZ

        elif self.activation_function == "relu":
            # GRADED FUNCTION: relu_backward
            ### START CODE HERE ###
            Z = self.cache
            dZ = dA * (Z > 0)
            ### END CODE HERE ###

            assert (dZ.shape == Z.shape)

            return dZ

        elif self.activation_function == "softmax":
            # GRADED FUNCTION: softmax_backward
            ### START CODE HERE ###
            Z = self.cache
            s = np.exp(Z - np.max(Z, axis=1, keepdims=True)) / np.sum(np.exp(Z - np.max(Z, axis=1, keepdims=True)), axis=1, keepdims=True)
            dZ = s - Y
            ### END CODE HERE ###

            assert (dZ.shape == self.cache.shape)

            return dZ

        elif self.activation_function == "linear":
            return dA

        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function}")

class Model():
    def __init__(self, units, activation_functions, loss_function):
        self.units = units
        self.activation_functions = activation_functions
        self.loss_function = loss_function
        self.initialize_parameters()

    def initialize_parameters(self):
        self.linear = []        # Store all Dense layers (weights & biases)
        self.activation = []    # Store all activation function layers

        for i in range(len(self.units)-1):
            dense = Dense(self.units[i], self.units[i+1], i)
            self.linear.append(dense)

        for i in range(len(self.activation_functions)):
            self.activation.append(Activation(self.activation_functions[i], self.loss_function))

    def forward(self, X):
        A = X

        # GRADED FUNCTION: model_forward
        ### START CODE HERE ###
        for i in range(len(self.linear)):
            A = self.linear[i].forward(A)
            A = self.activation[i].forward(A)
        ### END CODE HERE ###

        return A

    def backward(self, AL=None, Y=None):
        L = len(self.linear)
        C = Y.shape[1]

        # assertions
        warning = 'Warning: only the following 3 combinations are allowed! \n \
                    1. binary classification: sigmoid + cross_entropy \n \
                    2. multi-class classification: softmax + cross_entropy \n \
                    3. regression: linear + mse'
        assert self.loss_function in ["cross_entropy", "mse"], "you're using undefined loss function!"
        if self.loss_function == "cross_entropy":
            if Y.shape[1] == 1:  # binary classification
                assert self.activation_functions[-1] == 'sigmoid', warning
            else:  # multi-class classification
                assert self.activation_functions[-1] == 'softmax', warning
                assert self.units[-1] == Y.shape[1], f"you should set last dim to {Y.shape[1]}(the number of classes) in multi-class classification!"
        elif self.loss_function == "mse":
            assert self.activation_functions[-1] == 'linear', warning
            assert self.units[-1] == Y.shape[1], "output dimension mismatch for regression!"

        # GRADED FUNCTION: model_backward
        ### START CODE HERE ###
        if self.activation_functions[-1] == "linear":
            # Initializing the backpropagation
            dAL = AL - Y
            # Lth layer (LINEAR) gradients. Inputs: "dAL". Outputs: "dA_prev"
            dA_prev = self.linear[-1].backward(dZ=dAL)

        elif self.activation_functions[-1] == "sigmoid":
            # Initializing the backpropagation
            dAL = -(Y / (AL + 1e-5) - (1 - Y) / (1 - AL + 1e-5))

            # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL". Outputs: "dA_prev"
            dZ = self.activation[-1].backward(dA=dAL)
            dA_prev = self.linear[-1].backward(dZ=dZ)

        elif self.activation_functions[-1] == "softmax":
            # Initializing the backpropagation
            dZ = self.activation[-1].backward(Y=Y)

            # Lth layer (LINEAR) gradients. Inputs: "dZ". Outputs: "dA_prev"
            dA_prev = self.linear[-1].backward(dZ=dZ)

        # Loop from l=L-2 to l=0
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "dA_prev". Outputs: "dA_prev"
        for i in (range(L-2, -1, -1)):
            dZ = self.activation[i].backward(dA=dA_prev)
            dA_prev = self.linear[i].backward(dZ=dZ)
        ### END CODE HERE ###

        return dA_prev

    def update(self, learning_rate):
        L = len(self.linear)

        # GRADED FUNCTION: model_update_parameters
        ### START CODE HERE ###
        for i in range(L):
            self.linear[i].update(learning_rate)
        ### END CODE HERE ###
# GRADED FUNCTION: compute_BCE_loss

def compute_BCE_loss(AL, Y):
    n = Y.shape[0]

    # Compute loss from aL and y.
    ### START CODE HERE ### (≈ 1 line of code)
    loss = -(1/n) * np.sum(np.multiply(Y, np.log(AL + 1e-5)) + np.multiply(1 - Y, np.log(1 - AL + 1e-5)))
    ### END CODE HERE ###

    loss = np.squeeze(loss)      # To make sure your loss's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(loss.shape == ())

    return loss
# GRADED FUNCTION: compute_CCE_loss

def compute_CCE_loss(AL, Y):
    n = Y.shape[0]

    # Compute loss from aL and y.
    ### START CODE HERE ### (≈ 1 line of code)
    loss = -(1/n) * np.sum(np.multiply(Y, np.log(AL + 1e-5)))
    ### END CODE HERE ###

    loss = np.squeeze(loss)      # To make sure your loss's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(loss.shape == ())

    return loss
    
# compute_MSE_loss (MSE)
def compute_MSE_loss(AL, Y):
    m = Y.shape[0]
    loss = (1/m) * np.sum(np.square(AL - Y))
    return loss

def random_mini_batches(X, Y, mini_batch_size = 64):
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    ### START CODE HERE ###

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]

    # Step 2 - Partition (shuffled_X, shuffled_Y).
    # Cases with a complete mini batch size only i.e each of 64 examples.
    num_complete_minibatches = math.floor(m / mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : (k+1) * mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : (k+1) * mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # For handling the end case (last mini-batch < mini_batch_size i.e less than 64)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    ### END CODE HERE ###

    return mini_batches

def train_model(model, X_train, Y_train, learning_rate, num_iterations, batch_size=None, print_loss=True, print_freq=1000, decrease_freq=100, decrease_proportion=0.99):
    history = []
    losses = []

    for i in range(num_iterations):
        ### START CODE HERE ###
        # Define mini batches
        if batch_size:
            mini_batches = random_mini_batches(X_train, Y_train, batch_size)
        else:
            # if batch_size is None, batch is not used, mini_batch = whole dataset
            mini_batches = [(X_train, Y_train)]

        epoch_loss = 0
        for batch in mini_batches:
            X_batch, Y_batch = batch

            # Forward pass
            AL = model.forward(X_batch)

            # Compute loss
            if model.loss_function == 'cross_entropy':
                if model.activation_functions[-1] == "sigmoid": # Binary classification
                    loss = compute_BCE_loss(AL, Y_batch)
                elif model.activation_functions[-1] == "softmax": # Multi-class classification
                    loss = compute_CCE_loss(AL, Y_batch)
            elif model.loss_function == 'mse': # Regression
                loss = compute_MSE_loss(AL, Y_batch)
            epoch_loss += loss

            # Backward pass
            model.backward(AL, Y_batch)

            # Update parameters
            model.update(learning_rate)

        epoch_loss /= len(mini_batches)
        losses.append(epoch_loss)
        ### END CODE HERE ###

        # Print loss
        if print_loss and i % print_freq == 0:
            print(f"Loss after iteration {i}: {epoch_loss}")

        # Store history
        if i % 100 == 0:
            history.append((X_train, model.forward(X_train)))

        # Decrease learning rate
        if i % decrease_freq == 0 and i > 0:
            learning_rate *= decrease_proportion

    return model, losses, history
def predict(x, y_true, model):
    n = x.shape[0]

    # Forward propagation
    y_pred = model.forward(x)

    if y_pred.shape[-1] == 1:
        y_pred = np.array([[1 - y[0], y[0]] for y in y_pred])
        if y_true is not None:
            y_true = np.array([[1,0] if y == 0 else [0,1] for y in y_true.reshape(-1)])

    # make y_pred/y_true become one-hot prediction result
    if y_true is not None:
        y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    if y_true is not None:
        # compute accuracy
        correct = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == yp:
                correct += 1
        print(f"Accuracy: {correct/n * 100:.2f}%")

        f1_scores = f1_score(y_true, y_pred, average=None)
        print(f'f1 score for each class: {f1_scores}')
        print(f'f1_macro score: {np.mean(np.array(f1_scores)):.2f}')

    return y_pred

loss_function = 'cross_entropy'
layers_dims = [32400, 128, 64, 32, 1]
activation_fn = ['relu', 'relu', 'relu', 'sigmoid']
learning_rate = 0.01
num_iterations = 10
print_loss = True
print_freq = 1
decrease_freq = 10
decrease_proportion = 0.8
# You might need to use mini_batch to reduce training time in this part
batch_size = 32

model = Model(layers_dims, activation_fn, loss_function)
weight = np.load('NN_model.npy', allow_pickle=True)
j = 0
print(len(model.linear))
for w in weight:
    model.linear[j].parameters = w
    j = j + 1
test = np.array(read_images_from_folder('photo'), dtype="object")


print(test.shape)
test_label = np.zeros((test.shape[0], 1))
label1 = [0,1,2,3,4,5,6,7,11,12,13,14,15,16,17,18,21,26,27,28,29,30,31,32,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52]
for i in label1:
    test_label[i] = 1
pred_test = predict(test, test_label, model)
print(pred_test)