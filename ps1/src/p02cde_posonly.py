import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    #(c) use t
    x_train, y_train = util.load_dataset(train_path, label_col = 't', add_intercept=True)
    LR=LogisticRegression()
    LR.fit(x_train,y_train)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    y_pred = LR.predict(x_test)
    util.plot(x_test, y_test, LR.theta, '{}.png'.format(pred_path_c))
    np.savetxt(pred_path_c, y_pred)
    
    #(d) use y
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    LR=LogisticRegression()
    LR.fit(x_train,y_train)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    y_pred = LR.predict(x_test)
    util.plot(x_test, y_test, LR.theta, '{}.png'.format(pred_path_d))
    np.savetxt(pred_path_d, y_pred)
    
    #(e) use a held-out validation set to estimate alpha
    x_train, y_train = util.load_dataset(train_path, label_col = 'y', add_intercept=True)
    LR=LogisticRegression()
    LR.fit(x_train,y_train)
    x_valid = util.load_dataset(test_path, add_intercept=True)[0][util.load_dataset(test_path, add_intercept=True)[1]==1,:]
    alpha = np.mean(LR.predict(x_valid))
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    y_pred = LR.predict(x_test)/alpha
    LR.theta[0] = LR.theta[0]+np.log(2/alpha-1)
    util.plot(x_test, y_test, LR.theta, '{}.png'.format(pred_path_e))
    np.savetxt(pred_path_e, y_pred)
    # *** END CODER HERE
