import sys
import os
import numpy as np
import onnxruntime

# a SotA MNIST classfier that can actally judge 
# whether something is a four

# objective function needs to 
# 1. Accept points in latent space
# 2. decode these points into images
# 3. Throw them into the MNIST classifier
# 4. Accept the probability of a point being of digit X

# Solution:
# Let the objective function be a partial with everything but the point already being defined
# (Essentially, the partial becomes a class)
# Then, import objective from here and define a partial with all the stuff previously mentioned

def objective_function(point, decode, decoder_params, digit=0):

    X = np.array(decode(decoder_params, point).reshape(1, 1, 28, 28)) 
    onnx_session = onnxruntime.InferenceSession('mnist_pred.onnx')
    onnx_inputs = {onnx_session.get_inputs()[0].name : X}
    onnx_predictions = onnx_session.run(None, onnx_inputs)[0]
    return -onnx_predictions[:, digit][0]
    
# tries to find the prediction with the lowest variance (most uncertain pred)    
def brightest_item_objective_function(point, decode, decoder_params, digit=0, brightest=True):

    X = np.array(decode(decoder_params, point).reshape(1, 1, 28, 28)) 
    onnx_session = onnxruntime.InferenceSession('mnist_pred.onnx')
    onnx_inputs = {onnx_session.get_inputs()[0].name : X}
    onnx_predictions = onnx_session.run(None, onnx_inputs)[0]
    print(np.var(onnx_predictions), onnx_predictions)
    if brightest:
        return -onnx_predictions[:, digit][0] - np.sum(X)

    return -onnx_predictions[:, digit][0] + np.sum(X)