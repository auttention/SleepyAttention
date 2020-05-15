#!/usr/bin/python
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
import scipy.stats
import csv



# Define a function to map the predictions to 9-point KSS scale
def limit_kss(y):
    return np.clip(np.asarray(np.round(y), dtype=np.int32), 1, 9)


def baseline(train_features_path, devel_features_path, test_features_path, test_predictions_path, labels_path, num_feat):

    # Task
    #task_name = 'seq2seq/'
    #task_name = 'attention/'
    task_name = 'fusion/'

    # Configuration
    #feature_set = 'seq2seq_features'  # For all available options, see the dictionary feat_conf
    feature_set = 'attention_features'  # For all available options, see the dictionary feat_conf
    complexities = [1e-5,1e-4,1e-3,1e-2,1e-1,1e0]  # SVM complexities (linear kernel)


    # Mapping each available feature set to tuple (number of features, offset/index of first feature, separator, header option)
    feat_conf = {'attention_features': (256, 1, ',', None),
                 'seq2seq_features': (256, 1, ',', None),
                 'fusion_features': (512, 1, ',', None)}

    num_feat = int(num_feat)
    ind_off  = 1
    sep      = ','
    header   = None

    # Path of the features and labels
    #features_path = 'features/'
    #label_file    = 'labels/labels.csv'

    # Start
    print('\nRunning baseline... (this might take a while) \n')

    # Load features and labels
    X_train = pd.read_csv(train_features_path, sep=sep, header=header, usecols=range(ind_off, num_feat+ind_off), dtype=np.float32).values
    X_devel = pd.read_csv(devel_features_path, sep=sep, header=header, usecols=range(ind_off, num_feat+ind_off), dtype=np.float32).values
    X_test  = pd.read_csv(test_features_path,  sep=sep, header=header, usecols=range(ind_off, num_feat+ind_off), dtype=np.float32).values

    df_labels = pd.read_csv(labels_path)
    y_train = pd.to_numeric(df_labels['label'][df_labels['file_name'].str.startswith('train')]).values
    y_devel = pd.to_numeric(df_labels['label'][df_labels['file_name'].str.startswith('devel')]).values

    # Concatenate training and development for final training
    X_traindevel = np.concatenate((X_train, X_devel))
    y_traindevel = np.concatenate((y_train, y_devel))

    # Feature normalisation
    scaler       = MinMaxScaler()
    X_train      = scaler.fit_transform(X_train)
    X_devel      = scaler.transform(X_devel)
    X_traindevel = scaler.fit_transform(X_traindevel)
    X_test       = scaler.transform(X_test)

    # Train SVM model with different complexities and evaluate
    spearman_scores = []
    pearson_scores = []
    for comp in complexities:
        print('Complexity {0:.6f}'.format(comp))
        reg = svm.LinearSVR(C=comp, random_state=0)
        reg.fit(X_train, y_train)
        y_pred = limit_kss( reg.predict(X_devel))
        spearman = scipy.stats.spearmanr(y_devel, y_pred)[0]
        pearson  = scipy.stats.pearsonr(y_devel, y_pred)[0]
        if np.isnan(spearman):  # Might occur when the prediction is a constant
            spearman = 0.
            pearson = 0.
        spearman_scores.append( spearman )
        print('Spearman CC on Devel {0:.3f}'.format(spearman_scores[-1]))
        pearson_scores.append( pearson )
        print('Pearson CC on Devel {0:.3f}\n'.format(pearson_scores[-1]))

    # Train SVM model on the whole training data with optimum complexity and get predictions on test data
    optimum_complexity = complexities[np.argmax(spearman_scores)]
    print('Optimum complexity: {0:.6f}, maximum Spearman CC on Devel {1:.3f}\n'.format(optimum_complexity, np.max(spearman_scores)))

    reg = svm.LinearSVR(C=optimum_complexity, random_state=0)
    reg.fit(X_traindevel, y_traindevel)
    y_pred = limit_kss( reg.predict(X_test))

    # Write out predictions to csv file (official submission format)s
    print('Writing file ' + test_predictions_path + '\n')
    df = pd.DataFrame(data={'file_name': df_labels['file_name'][df_labels['file_name'].str.startswith('test')].values,
                            'prediction': y_pred.flatten()},
                      columns=['file_name', 'prediction'])
    os.makedirs(os.path.dirname(test_predictions_path), exist_ok=True)
    df.to_csv(test_predictions_path, index=False)

    print('Done.\n')


def eval_sleepiness(test_predictions_path, test_labels_path):
    y_pred = None
    df_labels = pd.read_csv(test_labels_path)
    y_test = pd.to_numeric( df_labels['label'][df_labels['file_name'].str.startswith('test')]).values
    # predictions
    if test_predictions_path.endswith('.csv'):  # otherwise it is a team name (fusion) and y_pred is given
        df_pred = pd.read_csv(test_predictions_path)
        y_pred = pd.to_numeric( df_pred['prediction'][df_pred['file_name'].str.startswith('test')]).values
    y_pred = limit_kss( y_pred )
    # score
    spearman_score = scipy.stats.spearmanr(y_test, y_pred)[0]
    print('SpearmanCC={0}'.format(spearman_score))
    return spearman_score


def fusion(seq2seq_features_path, attention_features_path, fusion_path):
    os.makedirs(os.path.dirname(fusion_path), exist_ok=True)
    seq2seq_features = pd.read_csv(seq2seq_features_path, header=None)
    attention_features = pd.read_csv(attention_features_path, header=None).iloc[:, 1:]
    fusion_features = pd.concat([seq2seq_features, attention_features], axis=1, ignore_index=True)
    fusion_features.to_csv(fusion_path, mode='w', index=False, header=False)