import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


def k_fold_CV(trainers, train_X, binary_y, real_y, k, l_rates=[]):
    ''' Performs the K-fold cross-validation for the 
        learning rate hyperparameter.

    Arguments:
        trainers (list): Trainers for 3 distinct Logistic
                         Regression models
        train_X (numpy.ndarray): train set input features
        binary_y (list): binary target output for each of
                         3 Logistic Regression models
        real_y (numpy.ndarray): Raw target output for the
                                train set
        k (int): number of folds
        l_rates (list/numpy.ndarray): possible values for
                                      the learning rate
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

    for lr in l_rates:
        val_metric = []
        train_metric = []
        for train_index, val_index in skf.split(train_X, real_y):
            for trainer in trainers:
                # Re-initialize the weights for each of
                # the Logistic Regression models
                trainer.model.init_weights()

            # Extract and standardize the train folds features
            train_features = train_X[train_index].copy()
            feat_mean = train_features.mean(axis=0, keepdims=True)
            feat_std = train_features.std(axis=0, keepdims=True)

            train_features = (train_features - feat_mean) / feat_std
            train_features = np.hstack([np.ones((len(train_features), 1)),
                                        train_features])

            # Extract and standardize the validation set features
            val_features = (train_X[val_index].copy() - feat_mean) / feat_std
            val_features = np.hstack([np.ones((len(val_features), 1)),
                                      val_features])

            y_hat_val = []
            y_hat_train = []
            for i, trainer in enumerate(trainers):
                # Form the train folds
                train_output = binary_y[i][train_index].copy()
                train_folds = [train_features, train_output]

                # Form the validation fold
                val_output = binary_y[i][val_index].copy()
                val_fold = [val_features, val_output]

                # Train the current Logistic Regression model
                trainer.train(train_folds, val_fold, lr)

                # Evaluate the model the train set
                y_hat = trainer.model.predict(train_features)
                y_hat_train.append(y_hat)

                # Evaluate the model on the validation set
                y_hat = trainer.model.predict(val_features)
                y_hat_val.append(y_hat)

            # Evaluate the ensemble on the validation set
            target_val = real_y[val_index]
            y_hat_val = np.argmax(y_hat_val, axis=0)
            val_acc = (y_hat_val == target_val).mean()
            val_metric.append(val_acc)

            # Evaluate the ensemble on the train set
            target_train = real_y[train_index]
            y_hat_train = np.argmax(y_hat_train, axis=0)
            train_acc = (y_hat_train == target_train).mean()
            train_metric.append(train_acc)

        # Calculate the necessary metrics
        history['val_mean'].append(np.mean(val_metric))
        history['val_std'].append(np.std(val_metric))
        history['train_mean'].append(np.mean(train_metric))
        history['train_std'].append(np.std(train_metric))

    for key in history.keys():
        history[key] = np.array(history[key])

    return history


class BinaryClassifier:
    ''' Represents Logistic Regression model
        used later for One-vs-All method. 
        A singular BinaryClassifier instance is
        used for classifying if an example belongs
        to the class which was attached to the model
        with @true_class argument in the constructor.
    '''

    def __init__(self, n_features, true_class):
        self.n_features = n_features
        self.true_class = true_class
        self.path = f"cls_{self.true_class}"
        self.init_weights()

    def predict(self, X):
        ''' Calculates the probability that
            samples X belong to the class that
            this instance represents.

        Arguments:
            X (numpy.ndarray): input features
        Returns:
            probs (numpy.ndarray): P(Y=1|X)
        '''
        out = X @ self.params
        probs = self.activation(out)
        return probs

    def activation(self, out):
        ''' Returns the sigmoid activation
            for the input argument.
        '''
        sigmoid = 1 / (1 + np.exp(-out))
        return sigmoid

    def save_state_dict(self, path=None):
        ''' Saves the model weights into a .npy file. '''
        if path is None:
            np.save(path, self.params)
        else:
            np.save(self.path, self.params)

    def load_state_dict(self, path=None):
        ''' Loads the model weights from the .npy file '''
        if path is None:
            self.params = np.load(path)
        else:
            self.params = np.load(self.path)

    def init_weights(self, init_type='xavier'):
        ''' Initializes the model weights. '''
        if init_type == 'xavier':
            # np.random.randn((n_features, 1))*np.sqrt(2/(fan_in+fan_out))
            self.params = np.random.randn(self.n_features, 1) * \
                np.sqrt(2 / (self.n_features + 1))


class TrainerLogistic:
    ''' Represents a trainer for a Logistic
        Regression model. 
    '''

    def __init__(self, model, epochs):
        self.model = model
        self.epochs = epochs

    def train(self, train_set, test_set=None, lr=1e-2, calc_loss=False):
        ''' Trains the model using batch Gradient Descent for a specified 
            number of epochs.

        Arguments:
            train_set (list): train set input features and target output
            test_set (list): val/test set input features and target output
            lr (float): learning rate
            calc_loss (bool): should the loss on the train set be calculated
        Returns:
            threshold (float): threshold which was used for evaluation
            train_loss (list): loss on the train set for each epoch
        '''
        x_train, y_train = train_set
        if test_set is not None:
            x_test, y_test = test_set

        mod = 50    # Print the train/val loss every mod epochs
        # The P(y=1|x) is compared to this threshold
        threshold = (y_train == 0).mean()

        train_loss = []
        for epoch in range(self.epochs):
            # Get the probability P(y=1|x)
            y_hat = self.model.predict(x_train)

            # Gradient of the sigmoid activation function
            sigmoid_grad = (1 - y_hat) * y_hat
            # Calculate the gradients in vectorized fashion
            gradients = x_train.T @ ((y_hat - y_train) * sigmoid_grad)
            # Update the model weights
            self.model.params -= lr * gradients

            if calc_loss:
                # Calculate the loss on the train set
                loss = (y_train * np.log(y_hat) + (1 - y_train)
                        * np.log(1 - y_hat)).sum()
                train_loss.append(loss)

            if test_set is not None and (epoch + 1) % mod == 0:
                # Evaluate the model performance on the val/test set
                y_hat = self.model.predict(x_test)
                y_hat = (y_hat >= threshold).astype('int')
                test_acc = (y_hat == y_test).mean()

        return threshold, train_loss
