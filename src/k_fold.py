import numpy as np


def k_fold_CV(model, features, target, k, orders=[], reg_lmbds=[]):
    ''' Performs the K-fold cross-validation for the given
        model and given hyperparameters. Before the k-fold
        an external function is called in order to split the
        dataset into k folds and couple them into train-val pairs.

    Arguments:
        model: model which we fit/evaluate
        features (numpy.ndarray): train set features
        target (numpy.ndarray): train set target output
        k (int): number of folds
        orders (list/numpy.ndarray): range of polynomial orders
                                     hyperparameter
        reg_lmbds (list/numpy.ndarray): list of regularization lambda
                                        hyperparameter
    '''
    assert ((len(orders)) or len(reg_lmbds)), "No hyperparameters given!"

    if not orders:
        orders = [2]
    else:
        metrics = dict()

    if not len(reg_lmbds):
        reg_lmbds = [0]

    for order in orders:
        mean_val = []
        mean_train = []
        std_val = []
        std_train = []

        for lmbd in reg_lmbds:
            val_losses = []
            train_losses = []

            # Get polynomial features for the raw features
            poly_features_train = get_poly_features(features, order)
            k_splits = k_fold_split(poly_features_train.copy(),
                                    target.copy(), k=k)

            for split in k_splits:
                # Standardize features
                standardized_features_train, feat_mean, feat_std = \
                    standardize_features(split['features_train'])
                # Padd the features with bias term: 1
                padded_features_train = np.hstack([np.ones((len(standardized_features_train), 1)),
                                                   standardized_features_train])

                standardized_features_val = (
                    split['features_val'] - feat_mean) / feat_std
                padded_features_val = np.hstack([np.ones((len(standardized_features_val), 1)),
                                                 standardized_features_val])

                if lmbd == 0:  # Check if no regularization should be applied
                    if isinstance(model, Ridge) or isinstance(model, Lasso):
                        model.fit(padded_features_train,
                                  split['target_train'], lmbd)
                    else:
                        model.fit(padded_features_train, split['target_train'])
                else:
                    model.fit(padded_features_train,
                              split['target_train'], lmbd)

                # Get the prediction for the validation set
                y_hat = model.predict(padded_features_val)
                loss_val = criterion(y_hat, split['target_val'])
                val_losses.append(loss_val)

                train_loss = criterion(model.predict(padded_features_train),
                                       split['target_train'])
                train_losses.append(train_loss)

            mean_val.append(np.mean(val_losses))
            mean_train.append(np.mean(train_losses))
            std_val.append(np.std(val_losses))
            std_train.append(np.std(train_losses))

        if len(orders) == 1:
            return np.array(mean_train), np.array(std_train), np.array(mean_val), np.array(std_val)
        else:
            # If we evaluate for more than one order then we return
            # a dict containing train/eval statistics for each order
            if len(reg_lmbds) == 1:
                metrics[str(order)] = mean_train[0], std_train[0], \
                    mean_val[0], std_val[0]
            else:
                metrics[str(order)] = [np.array(mean_train),
                                       np.array(std_train),
                                       np.array(mean_val),
                                       np.array(std_val)]

    return metrics


def k_fold_split(features, target, k=2):
    ''' Splits the input features and target
        outputs into k folds. An external function
        is called which couples those folds into
        a train validation pairs.

    Arguments:
        features (numpy.ndarray): input features
        target (numpy.ndarray): target output
        k (int): number of folds
    Returns:
        train_val_sets (list): list of train-val pairs
    '''
    assert k > 1, f"Invalid number of folds! Minimum number of folds is 2, but {k} is given!"
    dataset_len = len(features)

    # Length of one fold
    one_fold_len = np.ceil(dataset_len / k).astype('int')
    # Split the dataset into k folds
    folds_features = [features[i * one_fold_len:(i + 1) * one_fold_len]
                      for i in range(k)]
    folds_target = [target[i * one_fold_len:(i + 1) * one_fold_len]
                    for i in range(k)]

    # Form train-validation sets for each fold combination
    train_val_sets = create_train_val_sets(folds_features, folds_target)

    return train_val_sets


def create_train_val_sets(feature_folds, target_folds):
    ''' Couples the previously split folds into k
        different train-validation pairs.

    Arguments:
        feature_folds (list): k input features folds
        target_folds (list): k target output folds
    Returns:
        sets (list): k different train-val pairs
    '''
    sets = []
    k = len(feature_folds)

    for i in range(k):
        train_indices = list(range(k))
        del train_indices[i]

        # Create the current train set
        train_set_features = [feature_folds[ind] for ind in train_indices]
        train_set_target = [target_folds[ind] for ind in train_indices]

        curr_split = dict()
        curr_split['features_train'] = np.vstack(train_set_features).copy()
        curr_split['target_train'] = np.vstack(train_set_target).copy()
        # Couple the current validation set
        curr_split['features_val'] = feature_folds[i].copy()
        curr_split['target_val'] = target_folds[i].copy()

        sets.append(curr_split)

    return sets
