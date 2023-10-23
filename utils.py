import urllib
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from molfeat.trans import MoleculeTransformer
from sklearn.metrics import auc, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import selfies as sf
import numpy as np
import json
from sklearn import clone, metrics

import ml

import os


def download_data(filename):
    # Check if the file exists
    if not os.path.exists(filename):
        # If the file doesn't exist, download it
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/volkamerlab/teachopencadd/master/teachopencadd/talktorials/T002_compound_adme/data/EGFR_compounds_lipinski.csv",
            filename,
        )
    else:
        print(f"{filename} already exists in the current directory.")
    return pd.read_csv(filename).drop(columns=["Unnamed: 0"])


def smiles_to_descriptors(smiles, type=None):
    """
    Convert a SMILES string to various molecular descriptors.

    Parameters:
    - smiles (str): A SMILES string representing the chemical structure of a molecule.
    - type (str, optional): The type of descriptor to be calculated. Options are:
        - "mordred": Mordred-based molecular descriptors.
        - "maccs": MACCS keys.
        - "morgan2": Morgan fingerprint with a radius of 2.
        - "morgan3": Morgan fingerprint with a radius of 3.
        - "selfies": Selfies representation.
      If no type is specified or the specified type is not recognized, the function will return None.

    Returns:
    - desc (array-like): A numerical array representing the selected descriptor for the molecule.
      The output type and length depend on the selected descriptor.

    Requires:
    - rdkit: For generating the molecule object from SMILES and calculating MACCS and Morgan fingerprints.
    - MoleculeTransformer (if using mordred): For calculating Mordred-based descriptors.
    - selfies (if using selfies): For converting SMILES to selfies representation.

    Example:
    >>> smiles = "CCO"
    >>> maccs_descriptor = smiles_to_descriptors(smiles, type="maccs")
    """

    mol = Chem.MolFromSmiles(smiles)
    if type == "mordred":
        # Create Calculator for mordred features
        transformer = MoleculeTransformer(featurizer="mordred", dtype=float, n_jobs=8)
        desc = transformer(mol)
    elif type == "maccs":
        desc = np.array(MACCSkeys.GenMACCSKeys(mol))
    elif type == "morgan2":
        fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        desc = np.array(fpg.GetFingerprint(mol))
    elif type == "morgan3":
        fpg = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
        desc = np.array(fpg.GetFingerprint(mol))
    elif type == "selfies":
        desc = sf.encoder(smiles)
    else:
        raise ValueError(f"Descriptor type {type} not supported.")
    return desc


def model_training_and_validation(ml_model, name, splits, verbose=True):
    """
    Fit a machine learning model on a random train-test split of the data
    and return the performance measures.

    Parameters
    ----------
    ml_model: sklearn model object
        The machine learning model to train.
    name: str
        Name of machine learning algorithm: RF, SVM, ANN
    splits: list
        List of desciptor and label data: train_x, test_x, train_y, test_y.
    verbose: bool
        Print performance info (default = True)

    Returns
    -------
    tuple:
        Accuracy, sensitivity, specificity, auc on test set.

    """
    train_x, test_x, train_y, test_y = map(np.array, splits)

    # Fit the model
    ml_model.fit(train_x, train_y)

    # Calculate model performance results
    accuracy, auc, f1 = model_performance(ml_model, test_x, test_y, verbose)

    return accuracy, auc, f1


def model_performance(ml_model, test_x, test_y, verbose=True):
    """
    Helper function to calculate model performance

    Parameters
    ----------
    ml_model: sklearn model object
        The machine learning model to train.
    test_x: list
        Molecular fingerprints for test set.
    test_y: list
        Associated activity labels for test set.
    verbose: bool
        Print performance measure (default = True)

    Returns
    -------
    tuple:
        Accuracy, auc, f1 on test set.
    """

    # Prediction probability on test set
    test_prob = ml_model.predict_proba(test_x)[:, 1]

    # Prediction class on test set
    test_pred = ml_model.predict(test_x)

    # Performance of model on test set
    accuracy = accuracy_score(test_y, test_pred)
    auc = roc_auc_score(test_y, test_prob)
    f1 = f1_score(test_y, test_pred)

    if verbose:
        # Print performance results
        print(f"Accuracy: {accuracy:.2f}")
        print(f"AUC: {auc:.2f}")
        print(f"f1 score: {f1:.2f}")

    return accuracy, auc, f1


def selfies_to_encoding(selfies, vocab_stoi=None, pad_to_len=None):
    label = sf.selfies_to_encoding(
        selfies=selfies, vocab_stoi=vocab_stoi, pad_to_len=pad_to_len, enc_type="label"
    )
    return label


def prepare_RNN_data(
    X_train,
    X_val,
    X_test,
    input_shapes,
    selected_feature_indices=None,
    scale_descriptors=False,
):
    """
    Prepares data by slicing and scaling.

    Parameters:
    - X_train, X_val, X_test: input datasets
    - input_shapes: ist containing shapes [encoding_shape, fingerprint_shape(optional), descriptor_shape(optional)]
    - selected_feature_indices (optional): indices for feature selection. Note that if this is provided, descriptor_shape will be adjusted accordingly.
     This is if you know the indices to the best features to use among all features. If None, all are considered.
    - selected_feature_indices (optional): indices for feature selection.

    Returns:
    Sliced and scaled datasets.
    """

    encoding_shape = input_shapes[0]
    fingerprint_shape = input_shapes[1] if len(input_shapes) > 1 else 0
    descriptor_shape = input_shapes[2] if len(input_shapes) > 2 else 0

    outputs = []
    # Extract encoding input
    encoding_train = X_train[:, :encoding_shape]
    encoding_val = X_val[:, :encoding_shape]
    encoding_test = X_test[:, :encoding_shape]
    outputs.append((encoding_train, encoding_val, encoding_test))
    # Extract fingerprint if provided
    if fingerprint_shape > 0:
        fingerprint_train = X_train[
            :, encoding_shape : encoding_shape + fingerprint_shape
        ]
        fingerprint_val = X_val[:, encoding_shape : encoding_shape + fingerprint_shape]
        fingerprint_test = X_test[
            :, encoding_shape : encoding_shape + fingerprint_shape
        ]
        outputs.append((fingerprint_train, fingerprint_val, fingerprint_test))

    if descriptor_shape > 0:
        # Handle descriptor data
        descriptor_start_idx = encoding_shape + fingerprint_shape
        descriptor_train = X_train[:, descriptor_start_idx:]
        descriptor_val = X_val[:, descriptor_start_idx:]
        descriptor_test = X_test[:, descriptor_start_idx:]

        if selected_feature_indices is not None:
            descriptor_train = descriptor_train[:, selected_feature_indices]
            descriptor_val = descriptor_val[:, selected_feature_indices]
            descriptor_test = descriptor_test[:, selected_feature_indices]

        if scale_descriptors:
            # Scaling the descriptor inputs
            scaler = StandardScaler().fit(descriptor_train)
            descriptor_train = scaler.transform(descriptor_train)
            descriptor_val = scaler.transform(descriptor_val)
            descriptor_test = scaler.transform(descriptor_test)
        outputs.append((descriptor_train, descriptor_val, descriptor_test))

    return outputs


def base_model_crossvalidation(
    ml_model, df, X_columns, y_columns, n_folds=5, verbose=False, SEED=None
):
    """
    Cross validation wrapper for the baseline model

    Parameters
    ----------
    ml_model: sklearn model object
        The machine learning model to train.
    df: pd.DataFrame
        Data set with SMILES and their associated activity labels.
    X_columns: list
        List of column names for the input features.
    y_columns: list
        List of column names for the output label.
    n_folds: int, optional
        Number of folds for cross-validation.
    verbose: bool, optional
        Performance measures are printed.

    Returns
    -------
    None

    """
    # Shuffle the indices for the k-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=None)

    # Results for each of the cross-validation folds
    acc_per_fold = []
    f1_per_fold = []
    auc_per_fold = []

    # Loop over the folds
    for train_index, test_index in tqdm(kf.split(df)):
        # clone model -- we want a fresh copy per fold!
        fold_model = clone(ml_model)
        # Training
        # Convert the fingerprint and the label to a list
        train_x = np.vstack(df[X_columns].iloc[train_index])
        train_y = np.vstack(df[y_columns].iloc[train_index])

        # Fit the model
        fold_model.fit(train_x, train_y)

        # Testing

        # Convert the fingerprint and the label to a list
        test_x = np.vstack(df[X_columns].iloc[test_index])
        test_y = np.vstack(df[y_columns].iloc[test_index])

        # Performance for each fold
        accuracy, auc, f1 = model_performance(fold_model, test_x, test_y, verbose)

        # Save results
        acc_per_fold.append(accuracy)
        auc_per_fold.append(auc)
        f1_per_fold.append(f1)

    # Print statistics of results
    print(
        f"ACC:\t {np.mean(acc_per_fold):.2f}"
        f" ± {np.std(acc_per_fold):.2f} \n"
        f"AUC:\t {np.mean(auc_per_fold):.2f}"
        f" ± {np.std(auc_per_fold):.2f} \n"
        f"F1:\t {np.mean(f1_per_fold):.2f}"
        f" ± {np.std(f1_per_fold):.2f} \n"
    )
    results_dict = {
        "ACC": {
            "mean": f"{np.mean(acc_per_fold):.3g}",
            "std": f"{np.std(acc_per_fold):.3g}",
        },
        "AUC": {
            "mean": f"{np.mean(auc_per_fold):.3g}",
            "std": f"{np.std(auc_per_fold):.3g}",
        },
        "F1": {
            "mean": f"{np.mean(f1_per_fold):.3g}",
            "std": f"{np.std(f1_per_fold):.3g}",
        },
    }
    return results_dict


def get_features(s):
    """gets features from a smiles string s"""
    fingerprints = smiles_to_descriptors(s, type="morgan2")
    mordreds = smiles_to_descriptors(s, type="mordred")
    selfies = smiles_to_descriptors(s, type="selfies")

    with open("saved_results/non_zero_std_cols_mordred_indices.json", "r") as f:
        mordred_indices_dict = json.load(f)

    non_zero_std_cols = mordred_indices_dict["non_zero_std_cols"]
    optimal_mordred_features_indices = mordred_indices_dict[
        "optimal_mordred_features_indices"
    ]
    mordreds_non_nan = mordreds[:, non_zero_std_cols]
    mordreds_non_zero_std = mordreds_non_nan[:, optimal_mordred_features_indices]

    with open("saved_results/selfies_voc.json", "r") as f:
        voc = json.load(f)
    selfies_encoding = selfies_to_encoding(selfies, vocab_stoi=voc, pad_to_len=87)
    X_input_test = [
        np.array([selfies_encoding]),
        np.array([fingerprints]),
        mordreds_non_zero_std,
    ]
    return X_input_test


def RNN_model_crossvalidation(
    df,
    lstm_config,
    n_folds=5,
    verbose=False,
    optimal_mordred_features_indices=None,
    add_finger_print=False,
    add_mordred=False,
    SEED=None,
):
    """
    Cross validation warapper for RNN model

    """
    # Shuffle the indices for the k-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    # input_shapes = None

    # Results for each of the cross-validation folds
    acc_per_fold = []
    f1_per_fold = []
    auc_per_fold = []

    # Loop over the folds
    for train_index, test_index in tqdm(kf.split(df)):
        # Training

        mordred_features = np.vstack(df["mordred"])
        X_encoding = np.vstack(df["selfies encoding"])
        X_fingerprint = np.vstack(df["finger print"])
        X_mordred = mordred_features[:, optimal_mordred_features_indices]
        X = np.concatenate([X_encoding, X_fingerprint, X_mordred], axis=1)
        y = np.vstack(df["active"])

        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        splits = [X_train, X_test, y_train, y_test]

        if add_finger_print:
            input_shapes = [X_encoding.shape[1], X_fingerprint.shape[1]]
            outputs = prepare_RNN_data(X_train, X_test, X_test, input_shapes)
            X_train_encoding, X_val_encoding, X_test_encoding = outputs[0]
            X_train_fingerprint, X_val_fingerprint, X_test_fingerprint = outputs[1]
            X_train_input = [X_train_encoding, X_train_fingerprint]
            X_test_input = [X_test_encoding, X_test_fingerprint]

        if add_mordred:
            input_shapes = [
                X_encoding.shape[1],
                X_fingerprint.shape[1],
                X_mordred.shape[1],
            ]
            outputs = prepare_RNN_data(
                X_train, X_test, X_test, input_shapes, scale_descriptors=True
            )
            X_train_encoding, X_val_encoding, X_test_encoding = outputs[0]
            X_train_fingerprint, X_val_fingerprint, X_test_fingerprint = outputs[1]
            X_train_descriptor, X_val_descriptor, X_test_descriptor = outputs[2]
            X_train_input = [X_train_encoding, X_train_fingerprint, X_train_descriptor]
            X_test_input = [X_test_encoding, X_test_fingerprint, X_test_descriptor]

        if not add_finger_print and not add_mordred:
            input_shapes = [X_encoding.shape[1]]
            outputs = prepare_RNN_data(X_train, X_test, X_test, input_shapes)
            X_train_encoding, X_val_encoding, X_test_encoding = outputs[0]
            X_train_input = [X_train_encoding]
            X_test_input = [X_test_encoding]

        rnn_model = ml.RNNModel(lstm_config, input_shapes)
        result = rnn_model.train(
            X_train_input,
            y_train,
            validation_data=(X_test_input, y_test),
            verbose=verbose,
        )

        # Performance for each fold
        loss, accuracy, auc, f1 = rnn_model.evaluate(X_test_input, y_test)
        f1 = f1[0]

        # Save results
        acc_per_fold.append(accuracy)
        auc_per_fold.append(auc)
        f1_per_fold.append(f1)

    # Print statistics of results
    print(
        f"ACC:\t {np.mean(acc_per_fold):.2f}"
        f" ± {np.std(acc_per_fold):.2f} \n"
        f"AUC:\t {np.mean(auc_per_fold):.2f}"
        f" ± {np.std(auc_per_fold):.2f} \n"
        f"F1:\t {np.mean(f1_per_fold):.2f}"
        f" ± {np.std(f1_per_fold):.2f} \n"
    )
    results_dict = {
        "ACC": {
            "mean": f"{np.mean(acc_per_fold):.3g}",
            "std": f"{np.std(acc_per_fold):.3g}",
        },
        "AUC": {
            "mean": f"{np.mean(auc_per_fold):.3g}",
            "std": f"{np.std(auc_per_fold):.3g}",
        },
        "F1": {
            "mean": f"{np.mean(f1_per_fold):.3g}",
            "std": f"{np.std(f1_per_fold):.3g}",
        },
    }
    return results_dict
