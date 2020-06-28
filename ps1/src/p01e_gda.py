import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    model = GDA()
    model.fit(x_train, y_train)
    # Plot decision boundary on validation set
    x_val, y_val = util.load_dataset(eval_path, add_intercept=False)
    y_pred = model.predict(x_val)
    util.plot(x_val, y_val, model.theta, '{}.png'.format(pred_path))
    
    # Use np.savetxt to save outputs from validation set to pred_path
    np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***


class GDA(LinearModel):
    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        
        #MLE
        phi = sum(y==1)/len(y)
        mu_0 = sum(x[y==0,:])/sum(y==0)
        mu_1 = sum(x[y==1,:])/sum(y==1)
        mu = np.zeros((m,n))
        mu[y==1,:] = mu_1
        mu[y==0,:] = mu_0
        centered_x = x-mu
        sigma = 1/m*centered_x.T.dot(centered_x)
        
        inv_s = np.linalg.inv(sigma)
        theta = (mu_1-mu_0).dot(inv_s)
        theta0 = 1/2*(mu_0.dot(inv_s).dot(mu_0)-mu_1.dot(inv_s).dot(mu_1))-np.log((1-phi)/phi)
        
        self.theta = np.hstack([theta0, theta])
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        x = util.add_intercept(x)
        return x.dot(self.theta)>=0
        # *** END CODE HERE
