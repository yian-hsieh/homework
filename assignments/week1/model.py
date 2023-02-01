import numpy as np
import torch
from tqdm import trange

class LinearRegression:
    """
    A linear regression model that uses the closed form solution.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.b = 0
        self.w = np.zeros((1,1))
        return

    def fit(self, X: np.ndarray, y: np.ndarray,) -> None:
        """
        Fits model to given input and output.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The expected output.
        Returns:
            None
        """
        #bias_row = np.zeros((1, len(X[0])))
        #X = np.concatenate((bias_row, X), axis=0)
        #y = np.concatenate(([0], y), axis=0)
        params = np.linalg.inv(X.T @ X) @ (X.T @ y)
        self.b = 0
        self.w = params
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.
        """
        reg = X @ self.w
        #reg = self.w.T @ X + self.b
        return reg


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def _reg(self, X, w, b):
        X = torch.tensor(X).to(torch.float32)
        reg = torch.matmul(w.T, X) + b
        return reg

    def _mseloss(self, y_hat, y):
        y = torch.tensor(y).to(torch.float32)
        err = torch.mean(torch.square(y_hat - y))
        return err
    
    def _gradient_descent(self, w, b, lr):
        with torch.no_grad():
            w -= w.grad * lr
            b -= b.grad * lr
            w.grad.zero_()
            b.grad.zero_()
        return (w, b)
    
    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, 
            epochs: int = 1000) -> None:
        
        """
        Fits model to given input and output.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The expected output.
            lr (float): learning rate
            epochs(int): Epochs to train the model
        Returns:
            None
        """
        w = torch.zeros(X.shape[1], requires_grad=True)
        b = torch.zeros(X.shape[1], requires_grad=True)

        losses = []

        epoch_range = trange(epochs, desc="loss: ", leave=True)
        
        for epoch in epoch_range:
            if losses:
                epoch_range.set_description("loss: {:.6f}".format(losses[-1]))
                epoch_range.refresh()  # to show immediately the update

            y_hat = self._reg(X, w, b)
            l = self._mseloss(y_hat, y).mean()

            l.backward()
            w, b = self._gradient_descent(w, b, lr)  # Update parameters using their gradient

            losses.append(l)
        
        self.w = w
        self.b = b
        
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return self._reg(X, self.w, self.b).detach().numpy()
