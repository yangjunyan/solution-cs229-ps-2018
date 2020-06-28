import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    mse = []
    for tau in tau_values:
        LWR = LocallyWeightedLinearRegression(tau)
        LWR.fit(x_train, y_train)
        x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
        
        y_pred = LWR.predict(x_val)
        error = ((y_pred - y_val) ** 2).mean()
        mse.append(error)
        print('test mse: {}, tau:{}'.format(error,tau))
        
        #plot
        plt.figure()
        plt.plot(x_train[:,1:], y_train, 'bx')
        plt.plot(x_val[:,1:], y_pred, 'ro')
        plt.title('tau = %f'%(tau))
        plt.savefig('output/tau_{}.png'.format(tau))
    
    max_index = np.argsort(mse)[0]
    tau = tau_values[max_index]
    LWR = LocallyWeightedLinearRegression(tau)
    LWR.fit(x_train, y_train)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    y_pred = LWR.predict(x_test)
    test_mse = ((y_pred - y_test)**2).mean()
    np.savetxt(pred_path, y_pred)
    plt.figure()
    plt.title('$tau = {}$'.format(tau))
    plt.plot(x_train, y_train, 'bx')
    plt.plot(x_test, y_pred, 'ro')
    plt.savefig('output/final.png')
    print('lowest mse is %f, whose tau is %f, and mse on test set is %f.'%(mse[max_index], tau, test_mse))
    # *** END CODE HERE ***
