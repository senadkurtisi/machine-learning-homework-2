import matplotlib.pyplot as plt
import numpy as np


def transform_to_binary_output(output, num_classes=3):
    ''' Transforms the target output to the form below:
            Ex: for the class zero each example which 
                belongs to the class 0, is labeled as one
                while samples from classes 1 and 2 are
                labeled as 0. Similar process is performed
                for the other two classes.

        Target output in this form is later used for training
        the three separate logistic regression models.

    Arguments:
        output (numpy.ndarray): target output
        num_classes (int): number of distinct output classes
    Returns:
        binary (list): list of transformed target outputs
                       for each class
    '''
    binary = []
    for i in range(num_classes):
        # Which samples belong to the
        # current true class
        true_mask = output == i
        out_copy = output.copy()
        # Mark the samples belonging to
        # the current 'true class' as one
        out_copy[true_mask] = 1
        # Mark the samples belonging to the
        # other classes as 0
        out_copy[np.logical_not(true_mask)] = 0

        binary.append(out_copy)

    return binary


def show_plot(x_axis, history, title, xlabel, ylabel, scale='linear'):
    ''' Visualize the train and validation metrics.

    Arguments:
        x_axis (list/numpy.ndarray): values on the x axis of the plot
        history (dict): contains train/validation metrics
        xlabel (str): label of x axis
        ylabel (str): label of y axis
        scale (str): type of x-axis scale {log, linear}
    '''
    plt.figure()
    # Show the Train metric
    plt.plot(x_axis, history['train_mean'],
             color='darkorange',
             label='Training')
    plt.fill_between(x_axis, history['train_mean'] - history['train_std'],
                     history['train_mean'] + history['train_std'],
                     alpha=0.2,
                     color='darkorange')

    # Show the validation metric
    plt.plot(x_axis, history['val_mean'],
             label='Validation', color='navy')
    plt.fill_between(x_axis, history['val_mean'] - history['val_std'],
                     history['val_mean'] + history['val_std'],
                     alpha=0.2,
                     color='navy')

    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale(scale)
    plt.show()


def external_standardization(X_train, X_test):
    # Preprocess the training set
    feat_mean = X_train.mean(axis=0, keepdims=True)
    feat_std = X_train.std(axis=0, keepdims=True)
    X_train = (X_train.copy() - feat_mean) / feat_std
    X_train = np.hstack([np.ones((len(X_train), 1)),
                         X_train])

    # Preprocess the test set
    X_test = (X_test.copy() - feat_mean) / feat_std
    X_test = np.hstack([np.ones((len(X_test), 1)),
                        X_test])

    return X_train, X_test
