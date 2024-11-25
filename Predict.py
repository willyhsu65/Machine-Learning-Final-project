import numpy as np

# Helper function
def predict(model, X, y=None):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data that you would like to predict
    y -- the according label for the data
    model -- trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[0]
    n = len(model.layers) # number of layers in the neural network

    p = np.zeros((m,1))
    
    # Forward propagation
    probas = model.forward(X)
    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[0]):
        if probas[i] > 0.5:
            p[i,0] = 1
        else:
            p[i,0] = 0

    #print results
    if y is not None:
        print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p