import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    
    #train LR
    LR=LogisticRegression()
    LR.fit(x_train,y_train)

    #get prediction and decision boundary
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    y_pred = LR.predict(x_valid)
    util.plot(x_valid, y_valid, LR.theta, '{}.png'.format(pred_path))
    
    # Use np.savetxt to save predictions on eval set to pred_path
    np.savetxt(pred_path, y_pred)
    
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver."""

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        g = lambda x: 1/(1+np.exp(-x))
        m, n = x.shape
        
        if self.theta is None:
            self.theta = np.zeros(n)
            
        while True:
            theta = self.theta
            g_x = g(x.dot(self.theta))
            
            #Gradient
            G = -1/m*(y-g_x).dot(x)
            
            #Hessian
            H = 1/m*x.T.dot(np.diag(g_x*(1-g_x))).dot(x)
            self.theta = self.theta - np.linalg.inv(H).dot(G)
            if np.linalg.norm(self.theta-theta, ord=1) < self.eps:
                break    
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        g = lambda x: 1/(1+np.exp(-x))
        return g(x.dot(self.theta))
        # *** END CODE HERE ***
