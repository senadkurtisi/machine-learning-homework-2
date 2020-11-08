import numpy as np
from sklearn.model_selection import StratifiedKFold


class Softmax:
    ''' Represents a Softmax classifier '''

    def __init__(self, class_num, n_features):
        self.class_num = class_num
        self.n_features = n_features
        self.init_weights()

    def softmax(self, logits):
        ''' Performs the normalization of the exponents
            of logists.

        Arguments:
            logits (numpy.ndarray/float): logits
        Returns:
            exp_eta (numpy.ndarray/float): exponents of logist
                                           normalized to sum=1
        '''
        exp_eta = np.exp(logits)
        exp_eta = exp_eta / exp_eta.sum(axis=1, keepdims=True)

        return exp_eta

    def one_hot(self, target):
        ''' Creates a one hot vector representation
            for the output labels.

        Arguments:
            target (numpy.ndarray): target labesl
        Returns:
            oh_vec (numpy.ndarray): one-hot vector
                                    representation
        '''
        m = len(target)
        oh_vec = np.zeros((m, self.class_num))
        for i, output in enumerate(target):
            oh_vec[i, int(output)] = 1.0

        return oh_vec

    def predict(self, X):
        ''' Predicts the probabilities that X
            belongs to each of the possible 
            classes.

        Arguments:
            X (numpy.ndarray): input features
        Returns:
            probs (numpy.ndarray): probabilities
        '''
        logits = X @ self.params
        probs = self.softmax(logits)

        return probs

    def init_weights(self, init_type='xavier'):
        ''' Initializes the model weights. '''
        if init_type == 'xavier':
            self.params = np.random.randn(self.n_features, self.class_num) * \
                np.sqrt(2 / (self.class_num + self.n_features))


class TrainerSoftmax:
    ''' Represents the trainer for the Softmax model '''

    def __init__(self, model, epochs):
        self.model = model
        self.epochs = epochs

    def train(self, x_train, y_train, lr, bs=16, calc_loss=False):
        ''' Trains the model using Stochactic Mini-batch 
            Gradient Descent for a specified number of epochs.

        Arguments:
            x_train (numpy.ndarray): train set input features
            y_train (numpy.ndarray): train set labels
            lr (float): learning rate
            bs (int): mini-batch size
            calc_loss (bool): should the loss on the train set be calculated
        Returns:
            train_loss (list): loss on the train set for each epoch
        '''
        x_train = x_train.copy()
        y_train = y_train.copy()

        # Number of samples in the training set
        m = len(x_train)

        # Number of iterations through training set per epoch
        iters = m // bs
        if m % bs == 0:
            iters += 1

        train_loss = []

        for epoch in range(self.epochs):
            # Shuffle the samples in the training set
            indices = np.random.permutation(m)
            X_shuffled = x_train[indices]
            Y_shuffled = y_train[indices]

            for i in range(iters):
                # Extract the current mini-batch
                x = X_shuffled[i * bs:(i + 1) * bs]
                y = Y_shuffled[i * bs:(i + 1) * bs]

                # Get the one hot representation
                # of the labels
                y_one_hot = self.model.one_hot(y)
                # Get the prediction
                y_hat = self.model.predict(x)
                # Calculate the gradients
                gradients = x.T @ (y_hat - y_one_hot)

                # Update the model parameters
                self.model.params -= lr * gradients

                if calc_loss:
                    target_mat = np.zeros((len(y), 1))
                    for i, target in enumerate(y):
                        target_mat[i] = self.model.params[:,
                                                          int(target)] @ x[i].T

                    pred_mat = np.exp(x @ self.model.params)
                    pred_mat = np.log(pred_mat.sum(axis=1)).reshape((-1, 1))
                    loss = np.sum(target_mat - pred_mat) / len(y)

                    train_loss.append(loss)

        return train_loss


def k_fold_CV_bs(trainer, train_X, train_Y, k, lr, b_sizes=[]):
    ''' Performs the K-fold cross-validation for the 
        learning rate hyperparameter.

    Arguments:
        trainer (list): Trainer for Softmax model
        train_X (numpy.ndarray): train set input features
        train_y (list): train set labels
        k (int): number of folds
        lr (float): learning rate used for all models
        b_sizes (list/numpy.ndarray): possible values for
                                      the batch size
    Returns:
        history (dict): contains mean value and std. dev.
                        for the training and validation
                        accuracy
    '''
    skf = StratifiedKFold(n_splits=k, shuffle=False)

    history = {}
    history['val_mean'] = []
    history['val_std'] = []
    history['train_mean'] = []
    history['train_std'] = []

    for bs in b_sizes:
        val_metric = []
        train_metric = []
        for train_index, val_index in skf.split(train_X, train_Y):
            # Re-initialize the model weights
            trainer.model.init_weights()

            # Extract and standardize the train set features
            train_features = train_X[train_index].copy()
            feat_mean = train_features.mean(axis=0, keepdims=True)
            feat_std = train_features.std(axis=0, keepdims=True)
            train_features = (train_features - feat_mean) / feat_std
            train_features = np.hstack([np.ones((len(train_features), 1)),
                                        train_features])
            train_output = train_Y[train_index]

            # Train the Softmax classifier
            threshold = trainer.train(train_features, train_output, lr, bs)

            # Extract and standardize the validation fold features
            val_features = (train_X[val_index].copy() - feat_mean) / feat_std
            val_features = np.hstack([np.ones((len(val_features), 1)),
                                      val_features])
            val_output = train_Y[val_index]

            # Evaluate the model on the validation set
            y_hat_val = trainer.model.predict(val_features)
            y_hat_val = np.expand_dims(np.argmax(y_hat_val, axis=1), -1)
            val_acc = (y_hat_val == val_output).mean()
            val_metric.append(val_acc)

            # Evaluate the model on the test set
            y_hat_train = trainer.model.predict(train_features)
            y_hat_train = np.expand_dims(np.argmax(y_hat_train, axis=1), -1)
            train_acc = (y_hat_train == train_output).mean()
            train_metric.append(train_acc)

        # Calculate all necessary metrics
        history['val_mean'].append(np.mean(val_metric))
        history['val_std'].append(np.std(val_metric))
        history['train_mean'].append(np.mean(train_metric))
        history['train_std'].append(np.std(train_metric))

    for key in history.keys():
        history[key] = np.array(history[key])

    return history


def k_fold_CV_lr(trainer, train_X, train_Y, k, l_rates=[]):
    ''' Performs the K-fold cross-validation for the 
        learning rate hyperparameter.

    Arguments:
        trainer (list): Trainer for Softmax model
        train_X (numpy.ndarray): train set input features
        train_y (list): train set labels
        k (int): number of folds
        l_rates (list/numpy.ndarray): possible values for
                                      the learning rate
    Returns:
        history (dict): contains mean value and std. dev.
                        for the training and validation
                        accuracy
    '''
    skf = StratifiedKFold(n_splits=k, shuffle=False)

    history = dict()
    history['val_mean'] = list()
    history['val_std'] = list()
    history['train_mean'] = list()
    history['train_std'] = list()

    for lr in l_rates:
        val_metric = list()
        train_metric = list()
        for train_index, val_index in skf.split(train_X, train_Y):
            # Re-initialize the model weights
            trainer.model.init_weights()

            train_features = train_X[train_index].copy()
            feat_mean = train_features.mean(axis=0, keepdims=True)
            feat_std = train_features.std(axis=0, keepdims=True)
            # Standardize the input features
            train_features = (train_features - feat_mean) / feat_std
            train_features = np.hstack([np.ones((len(train_features), 1)),
                                        train_features])
            train_output = train_Y[train_index]

            # Train the Softmax classifier
            threshold = trainer.train(train_features, train_output, lr)

            # Standardize the validation set features
            val_features = (train_X[val_index].copy() - feat_mean) / feat_std
            val_features = np.hstack([np.ones((len(val_features), 1)),
                                      val_features])
            val_output = train_Y[val_index]

            # Evaluate the model on the test set
            y_hat_val = trainer.model.predict(val_features)
            y_hat_val = np.expand_dims(np.argmax(y_hat_val, axis=1), -1)
            val_acc = (y_hat_val == val_output).mean()
            val_metric.append(val_acc)

            # Evaluate the model on the train set
            y_hat_train = trainer.model.predict(train_features)
            y_hat_train = np.expand_dims(np.argmax(y_hat_train, axis=1), -1)
            train_acc = (y_hat_train == train_output).mean()
            train_metric.append(train_acc)

        # Calulate the necessary metrics
        history['val_mean'].append(np.mean(val_metric))
        history['val_std'].append(np.std(val_metric))
        history['train_mean'].append(np.mean(train_metric))
        history['train_std'].append(np.std(train_metric))

    for key in history.keys():
        history[key] = np.array(history[key])

    return history
