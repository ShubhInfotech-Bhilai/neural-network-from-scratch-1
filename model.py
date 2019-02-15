import numpy as np


class NeuralNetwork:
    def __init__(self, n_nodes, input_dim, lr, seed=1749):
        self.lr = lr
        self.n_nodes = n_nodes
        self.seed = seed  # Random seed for weight initialization

        # Initialize the first layer
        self.W1 = self.init_weights(n_nodes=self.n_nodes, input_dim=input_dim)
        self.b1 = self.init_bias(n_nodes=self.n_nodes)

        # Initialize the second layer
        self.W2 = self.init_weights(n_nodes=1, input_dim=self.n_nodes)
        self.b2 = self.init_bias(n_nodes=1)

    def init_weights(self, n_nodes, input_dim):
        """
        Each row is a node, each column is a dimension of the input,
        so dimension is (n_nodes, input_dim).
        We initialize with a random normal in order to break symmetry.
        """
        np.random.seed(self.seed)
        W = np.random.normal(size=(n_nodes, input_dim))
        return W

    def init_bias(self, n_nodes):
        """
        The bias is added to Wx, therefore has dimension (n_nodes, 1).
        Bias can be initialized to 0 since it isn't affected by the problem
        of symmetry.
        """
        np.random.seed(self.seed)
        b = np.zeros((n_nodes, 1))
        return b

    def sigmoid(self, z):
        """
        This is the activation function. It takes as input
        z = Wx + b, and applies the sigmoid function.
        """
        return 1/(1 + np.exp(-z))

    def deriv_sigmoid(self, z):
        """
        The derivation of the sigmoid function
        """
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def compute_loss(self, y_hat, y, offset=True):
        '''
        Here we compute the loss function between two predictions.
        If offset is true, we use a small value of epsilon to avoid numerical
        problems.
        '''
        if offset:
            epsilon = 10e-30
        else:
            epsilon = 0

        loss = - (y * np.log(y_hat + epsilon) + (1-y) * np.log(1 - y_hat + epsilon))
        return np.average(loss)

    def forward_prop(self, W1, b1, W2, b2, X, y):
        '''
        This process the forward propagation step
        '''
        # First hidden layer
        Z1 = W1.dot(X) + b1
        A1 = self.sigmoid(Z1)

        # 2nd layer (output layer)
        Z2 = W2.dot(A1) + b2
        A2 = self.sigmoid(Z2)

        return A1, A2, Z1, Z2

    def backprop(self, A1, A2, W1, W2, Z1, Z2, X, y):
        """
        This process the backward propagation algorithm.
        """
        m = X.shape[1]

        # Compute derivatives for second (output) layer
        dZ2 = A2 - y
        dW2 = (1/m) * dZ2.dot(A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        # Computer derivatives for first layer
        dZ1 = W2.T.dot(dZ2) * self.deriv_sigmoid(Z1)
        dW1 = (1/m) * dZ1.dot(X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        return dW1, dW2, db1, db2

    def update_weights(self, lr, dW1, dW2, db1, db2):
        """
        Updates all the weights in place
        """
        # Update all the layers
        self.W1 = self.W1 - lr * dW1
        self.W2 = self.W2 - lr * dW2
        self.b1 = self.b1 - lr * db1
        self.b2 = self.b2 - lr * db2

    def dropout(self, A, keep_prob=0.8):
        '''Method used for regularization.'''
        np.random.seed(self.seed)
        D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
        out = np.multiply(A, D)
        out /= keep_prob
        return out

    def train(self, X, y, n_iter=200, display_training=True):
        # Run for n_iter iterations
        for n in range(1, n_iter+1):
            A1, A2, Z1, Z2 = self.forward_prop(
                self.W1, self.b1, self.W2, self.b2, X, y
            )

            A1 = self.dropout(A1)

            dW1, dW2, db1, db2 = self.backprop(
                A1, A2, self.W1, self.W2, Z1, Z2, X, y
            )

            self.update_weights(self.lr, dW1, dW2, db1, db2)

            # Display loss
            if n % 50 == 0 and display_training:
                loss = self.compute_loss(A2, y)
                print(f"Iteration: {n}.\t Loss: {loss}")

    def predict(self, X):
        # First hidden layer
        Z1 = self.W1.dot(X) + self.b1
        A1 = self.sigmoid(Z1)

        # 2nd layer (output layer)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.sigmoid(Z2)

        y_proba = np.squeeze(A2)
        y_pred = np.around(y_proba)

        return y_pred, y_proba

    def compute_accuracy(self, y_pred, y):
        return np.average(y_pred == y)
