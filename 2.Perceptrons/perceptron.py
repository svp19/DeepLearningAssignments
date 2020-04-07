import numpy as np

class Perceptron():

    def __init__(self, X, Y):
        # Append bias to input X
        bias = 1 * np.ones((X.shape[0], 1))
        self.X = np.append(bias, X, axis=1)
        self.Y = Y
        self.w = np.random.random((1, self.X.shape[1]))
    
    def weighted_sum(self, x):
        return np.dot(self.w, x)
    
    def unit_step(self, x):
        if x < 0:
            return 0
        return 1
    
    
    def train(self, eta=0.5, epochs=10, method="pll"):
        
        for e in range(epochs):
            
            error = 0
            input_order = np.random.permutation(self.X.shape[0])
            for i in input_order:
                
                x = self.X[i]
                output = self.unit_step(self.weighted_sum(x))
                
                # Perceptron Learning Law
                if method == "pll":
                    if output <= 0 and self.Y[i] == 1:
                        error += 1
                        self.w = self.w + eta * x

                    elif output > 0 and self.Y[i] == 0:
                        error += 1
                        self.w = self.w - eta * x
                
                # Gradient Descent
                if method == "gd":
                    error += np.abs(self.Y[i] - output)
                    self.w = self.w + eta * (self.Y[i] - output) * x 
                    

            print(f"Epoch: {e}, Error: {error}, updated_Weights: {self.w}")

X = np.array([
    [0, 0], 
    [0, 1],
    [1, 0],
    [1, 1]    
])

Y = np.array([0, 1, 1, 1])

p = Perceptron(X, Y)

p.train(method="pll")
# p.train(method="gd")