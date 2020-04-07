import numpy as np

#Weights
w0 = 2*np.random.random((2, 5)) - 1 #for input   - 4 inputs, 5s outputs
w1 = 2*np.random.random((5, 1)) - 1 #for layer 1 - 5 inputs, 3 outputs

#learning rate
n = 0.5

#Errors - for graph later
errors = []

#sigmoid and its derivative
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x)*(1 - sigmoid(x))

#Normalize array
def normalize(X, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

X_train =np.transpose( [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0]])
Y_train = np.transpose([ [0], [1], [1], [0]])

#Train
for i in range(1000):

    #Feed forward
    layer0 = X_train
    print(layer0.shape, w0.shape)
    layer1 = sigmoid(np.dot(layer0.T, w0))
    print(layer1.shape)
    layer2= sigmoid(np.dot(layer1, w1))
    layer2 = layer2.T
    print(layer2.shape)
    print(Y_train.shape)
    #Back propagation using gradient descent
    layer2_error = Y_train - layer2
    print(sigmoid_deriv(layer2).shape)
    layer2_delta = layer2_error * sigmoid_deriv(layer2)
    
    
    layer1_error = layer2_delta.T.dot(w1.T)
    layer1_delta = layer1_error * sigmoid_deriv(layer1)
    
    w1 += layer1.T.dot(layer2_delta.T) * n
    w0 += layer0.dot(layer1_delta) * n
    print(w0, w1)
    
    error = np.mean(np.abs(layer2_error))
    errors.append(error)
    accuracy = (1 - error) * 100
    print(accuracy)

# #Plot the accuracy chart
# plt.plot(errors)
# plt.xlabel('Training')
# plt.ylabel('Error')
# plt.show()
        
# print("Training Accuracy " + str(round(accuracy,2)) + "%")

# layer0 = X_test
# layer1 = sigmoid(np.dot(layer0, w0))
# layer2 = sigmoid(np.dot(layer1, w1))

# layer2_error = y_test - layer2

# error = np.mean(np.abs(layer2_error))
# accuracy = (1 - error) * 100