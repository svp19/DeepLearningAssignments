import numpy as np

#sigmoid 
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_dash(x):
    return sigmoid(x)*(1 - sigmoid(x))


class NeuralNetwork:

    def __init__(self):
        self.W = -1
    
    def computation(self, A, B):
        # W = BA+
        print(A.shape)
        print(B.shape)
        self.L = B.shape[1]
        self.W = np.matmul(B, np.linalg.pinv(A))
        self.E = (1/self.L) * np.trace(  np.matmul( np.matmul(B, np.identity(self.L)), np.transpose(B)) )
    
    def widrow(self, A, B, learning_rate, num_epochs):
        self.W = np.zeros((B.shape[0], A.shape[0]))
        M = A.shape[0]
        N = B.shape[0]
        for l in range(num_epochs):
            delta = np.matmul(self.W, A) - B
            self.W = self.W - learning_rate * np.matmul(delta, A.T)

    def hebbian(self, A, B):
        self.W = np.matmul(B, np.transpose(A))

    def predict(self, A):
        return np.matmul(self.W, A)

    
nn = NeuralNetwork()
A = np.array([[0,0], [0,1], [1,0], [1,1]]).T
B = np.array([ [0], [1], [1], [1] ]).T

print(A.shape)
print(B.shape)


nn.computation(A, B, 0.01, 4)
print(nn.predict(A))