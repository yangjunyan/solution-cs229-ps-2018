import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    PR = PoissonRegression(max_iter = 1e5,step_size = lr, eps = 4e-9)
    PR.fit(x_train, y_train)
    x_test, y_test = util.load_dataset(train_path, add_intercept=False)
    y_pred = PR.predict(x_test)
    np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m,n = x.shape
        j = 0
        if self.theta is None:
            self.theta = np.zeros(n)
        
        #using SGD
        while True:
            for i in range(m):
                j = j+1
                theta = self.theta
                grad = (y[i] - np.exp(x[i,:].dot(theta)))*x[i,:]
                self.theta = theta + self.step_size * grad
                print(theta)
                if np.linalg.norm(self.theta-theta, ord=1)<self.eps or j>self.max_iter:
                    break
            else:
                continue
            break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x.dot(self.theta))
        # *** END CODE HERE ***
