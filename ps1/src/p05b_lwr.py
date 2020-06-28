import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    LWR = LocallyWeightedLinearRegression(0.5)
    LWR.fit(x_train, y_train)
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    y_pred = LWR.predict(x_val)
    
    #plot
    plt.figure()
    plt.plot(x_train[:,1:], y_train, 'bx')
    plt.plot(x_val[:,1:], y_pred, 'ro')
    plt.show()
    
    mse = ((y_pred - y_val) ** 2).mean()
    print(mse)
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m,n = self.x.shape
        y_pred = []
        for t in x:
            # calculate matrix omiga
            omiga = np.diag(np.exp(-np.sum((self.x - np.tile(t,(m,1)))**2/(2*self.tau**2), axis = 1)))
            
            #prediction
            y_pred.append(t.dot(np.linalg.inv((self.x.T.dot(omiga).dot(self.x)))).dot(self.x.T).dot(omiga).dot(self.y))
        return y_pred
        # *** END CODE HERE ***
