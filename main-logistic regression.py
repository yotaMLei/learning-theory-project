"""
Introduction to Machine Learning - Programming Assignment
Exercise 04
December 2020
Yotam Leibovitz
"""
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import time


def tic():
    return time.time()


def toc(t):
    return float(tic()) - float(t)

#########################################################################
#                Load Dataset                                           #
#########################################################################
mnist = fetch_openml('mnist_784')
X_src = mnist['data'].astype('float64')
t_src = mnist['target']

# Split the data into 3 different binary classification problems
index_01 = (t_src == '0') | (t_src == '1')
t_01 = t_src[index_01]    # target labels of the digit '0' or '1' only
X_01 = X_src[index_01]    # data samples of the digit '0' or '1' only

index_23 = (t_src == '2') | (t_src == '3')
t_23 = t_src[index_23]    # target labels of the digit '2' or '3' only
X_23 = X_src[index_23]    # data samples of the digit '2' or '3' only

index_45 = (t_src == '4') | (t_src == '5')
t_45 = t_src[index_45]    # target labels of the digit '4' or '5' only
X_45 = X_src[index_45]    # data samples of the digit '4' or '5' only

#convert labels to binary {0, 1} (for logisic regression)
t_01[t_01 == '0'] = 0
t_01[t_01 == '1'] = 1

t_23[t_23 == '2'] = 0
t_23[t_23 == '3'] = 1

t_45[t_45 == '4'] = 0
t_45[t_45 == '5'] = 1


# train test split

# shuffle the samples
# t_list = [t_0_1_only, t_2_3_only, t_4_5_only]
# X_list = [X_0_1_only, X_2_3_only, X_4_5_only]

random_state = check_random_state(1)
permutation = random_state.permutation(X_01.shape[0])
X_01 = X_01[permutation]
t_01 = t_01[permutation]
X_01 = X_01.reshape((X_01.shape[0], -1))  # This line flattens the image into a vector of size 784
X_01_train, X_01_test, t_01_train, t_01_test = train_test_split(X_01, t_01, test_size=0.3)

random_state = check_random_state(1)
permutation = random_state.permutation(X_23.shape[0])
X_23 = X_23[permutation]
t_23 = t_23[permutation]
X_23 = X_23.reshape((X_23.shape[0], -1))  # This line flattens the image into a vector of size 784
X_23_train, X_23_test, t_23_train, t_23_test = train_test_split(X_23, t_23, test_size=0.3)

random_state = check_random_state(1)
permutation = random_state.permutation(X_45.shape[0])
X_45 = X_45[permutation]
t_45 = t_45[permutation]
X_45 = X_45.reshape((X_45.shape[0], -1))  # This line flattens the image into a vector of size 784
X_45_train, X_45_test, t_45_train, t_45_test = train_test_split(X_45, t_45, test_size=0.3)

# split again to obtain validation set (20% validation, 20% test)
# X_validation, X_test, t_validation, t_test = train_test_split(X_test, t_test, test_size=0.5)

# plt.imshow(X_2_3_only[0].reshape((28,28)))

#########################################################################
#                Preprocessing                                          #
#########################################################################
# The next lines standardize the images
scaler = StandardScaler() # translate and scale sample vectors to zero mean and unit variance  #TODO: try also using MaxAbsScaler instead

# print(np.mean(X_01), np.std(X_01))
# print(np.mean(X_23), np.std(X_23))
# print(np.mean(X_45), np.std(X_45))

# standardize full data sets
X_01 = scaler.fit_transform(X_01)
X_23 = scaler.fit_transform(X_23)
X_45 = scaler.fit_transform(X_45)

# print(np.mean(X_01), np.std(X_01))
# print(np.mean(X_23), np.std(X_23))
# print(np.mean(X_45), np.std(X_45))

# standardize train-test data sets
X_01_train = scaler.fit_transform(X_01_train)
X_01_test = scaler.transform(X_01_test) # translate and scale the test data with the train mean variance values
X_23_train = scaler.fit_transform(X_23_train)
X_23_test = scaler.transform(X_23_test) # translate and scale the test data with the train mean variance values
X_45_train = scaler.fit_transform(X_45_train)
X_45_test = scaler.transform(X_45_test) # translate and scale the test data with the train mean variance values

# add 1 at the of every sample for the bias term
X_01 = np.hstack((X_01, np.ones((X_01.shape[0], 1))))
X_23 = np.hstack((X_23, np.ones((X_23.shape[0], 1))))
X_45 = np.hstack((X_45, np.ones((X_45.shape[0], 1))))

X_01_train = np.hstack((X_01_train, np.ones((X_01_train.shape[0], 1))))
X_01_test = np.hstack((X_01_test, np.ones((X_01_test.shape[0], 1))))
X_23_train = np.hstack((X_23_train, np.ones((X_23_train.shape[0], 1))))
X_23_test = np.hstack((X_23_test, np.ones((X_23_test.shape[0], 1))))
X_45_train = np.hstack((X_45_train, np.ones((X_45_train.shape[0], 1))))
X_45_test = np.hstack((X_45_test, np.ones((X_45_test.shape[0], 1))))

# convert from chars to int
t_01 = t_01.astype(int)
t_23 = t_23.astype(int)
t_45 = t_45.astype(int)

t_01_train = t_01_train.astype(int)
t_01_test = t_01_test.astype(int)
t_23_train = t_23_train.astype(int)
t_23_test = t_23_test.astype(int)
t_45_train = t_45_train.astype(int)
t_45_test = t_45_test.astype(int)

# create initial random weights
W = np.random.random((785, 1))

#########################################################################
#                Auxiliary Functions                                    #
#########################################################################
def compute_Y(X, t, W):
    A = X @ W
    Y = np.zeros((X.shape[0], 10))
    for i in range(A.shape[0]):
        # prevent overflow by subtracting the max value from each entry in row i
        A[i, :] = A[i, :] - A[i, :].max()
        A[i, :] = np.exp(A[i, :])
        Y[i, :] = A[i, :] / A[i, :].sum()
    return Y


def compute_loss(Y, t):
    y_true = Y[np.arange(0, Y.shape[0], 1), t]
    # set minimum value so log() function won't overflow
    y_true[y_true < 1e-100] = 1e-100
    return np.sum(-1 * np.log(y_true))


def compute_gradient(Y, X, t):
    T = np.zeros((X.shape[0], 10))
    T[np.arange(0, X.shape[0], 1), t] = 1  # t_n_k = the (n,k) entry of matrix T
    gradients = [(X.T @ ((Y - T)[:, [j]])) for j in range(10)]
    grad_E = np.hstack(gradients)
    return grad_E


def compute_accuracy(X, t, W):
    T = np.zeros((X.shape[0], 10))
    T[np.arange(0, X.shape[0], 1), t] = 1  # t_n_k = the (n,k) entry of matrix T
    Y = compute_Y(X, t, W)
    for row in Y[:, :]:
        row[row < row.max()] = -1
        row[row == row.max()] = 1
    correct_classifications = (Y == T).sum()
    accuracy = correct_classifications / X.shape[0]
    return accuracy


def step(X, t, Y, W, step_size):
    grad_E = compute_gradient(Y, X, t)
    W_new = W - step_size * grad_E
    return W_new


def GD_optimizer(X_train, t_train, X_validation, t_validation, X_test, t_test, W, step_size, threshold):
    step_num = 0
    Y_train = compute_Y(X_train, t_train, W)
    loss_list = np.array(())
    val_accuracy_list = np.array(())
    recent_mean_accuracy_diff = 100  # set initial accuracy difference to 100
    # gradient descent
    while np.abs(recent_mean_accuracy_diff) >= threshold:
        step_num += 1
        # update weights
        W = step(X_train, t_train, Y_train, W, step_size)
        # compute new loss
        Y_train = compute_Y(X_train, t_train, W)
        loss_list = np.append(loss_list, compute_loss(Y_train, t_train))
        val_accuracy_list = np.append(val_accuracy_list, compute_accuracy(X_validation, t_validation, W))
        if step_num > 3:
            # compute mean of accuracy difference for the last 3 steps
            recent_mean_accuracy_diff = np.mean(val_accuracy_list[-3:] - val_accuracy_list[-4:-1])

    # compute final accuracy
    train_acc = compute_accuracy(X_train, t_train, W)
    test_acc = compute_accuracy(X_test, t_test, W)
    validation_acc = val_accuracy_list[-1]
    return train_acc, validation_acc, test_acc, loss_list, val_accuracy_list, step_num


#########################################################################
#                Main                                                   #
#########################################################################
if __name__ == "__main__":
    # initialize hyper-parameters
    step_size = 0.05
    threshold = 0.0005

    # run the gradient descent optimizer
    train_acc, validation_acc, test_acc, loss_list, val_accuracy_list, step_num = GD_optimizer(
        X_train,
        t_train,
        X_validation,
        t_validation,
        X_test, t_test,
        W, step_size,
        threshold)

    print(
        "**** MNIST Multiclass Logistic Regression Results ****\nHyper-parameters :\n       step size = {4}\n"
        "       threshold = {5}\nNumber of steps until convergence = {0} steps"
        "\nFinal training accuracy: {1} \nFinal validation accuracy: {2} \nFinal test accuracy: {3}\n\n\n".format(
            step_num, train_acc, validation_acc, test_acc, step_size, threshold))

    # plot the results
    x = np.arange(1, step_num + 1, 1)
    fig = plt.figure()

    ax = fig.add_subplot(211)
    ax.set(title='Loss per iteration', xlabel='Iteration', ylabel='E(W)')
    ax.plot(x, loss_list, linewidth=2, c='C3', marker='o')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    ax = fig.add_subplot(212)
    ax.plot(x, 100*val_accuracy_list, linewidth=2, c='C3', marker='o')
    ax.set(title='Validation accuracy per iteration', xlabel='Iteration', ylabel='Accuracy')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.tight_layout()
    plt.savefig('ML_ex4_plots.png')
    plt.show()

