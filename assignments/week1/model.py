import numpy as np


class LinearRegression:

    w: np.ndarray
    b: float

    def __init__(self):

    def fit(self, X, y):
        params = np.linalg.inv(X.T @ X) @ (X.T @ y)
        self.w = params[-1]
        self.b = params[:-1]

    def predict(self, X):
        reg = torch.matmul(torch.transpose(self.w, 0, -1), X) + self.b
        return reg


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def linear_regression(X, w, b):
        reg = torch.matmul(torch.transpose(w, 0, -1), X) + b
        return reg

    def mseloss(y_hat, y):
        err = torch.mean(torch.square(y_hat - y))
        return err
    
    def gradient_descent(w, b, lr):
        with torch.no_grad():
            w -= w.grad * lr
            b -= b.grad * lr
            w.grad.zero_()
            b.grad.zero_()
        return (w, b)
    
    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, 
            epochs: int = 1000) -> None:
        w = torch.zeros(1, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)

        losses = []

        epoch_range = trange(num_epochs, desc="loss: ", leave=True)
        
        for epoch in epoch_range:
            if losses:
                epoch_range.set_description("loss: {:.6f}".format(losses[-1]))
                epoch_range.refresh()  # to show immediately the update

            y_hat = linear_regression(X, w, b)
            l = mseloss(y_hat, y_train).mean()

            l.backward()
            w, b = gradient_descent(w, b, lr)  # Update parameters using their gradient

            losses.append(l)

            time.sleep(0.01)
        
        self.w = w
        self.b = b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return linear_regression(X, self.w, self.b)
