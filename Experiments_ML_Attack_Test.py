# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 14:48:52 2021

@author: u0138175
"""
import random
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import roc_curve
# import packages
# packages from https://github.com/irhete/predictive-monitoring-benchmark/blob/master/experiments/experiments.py
from DatasetManager import DatasetManager
import EncoderFactory
import warnings

warnings.filterwarnings("ignore")

# REMOVE ERROR PRINTS###################
pd.options.mode.chained_assignment = None


# create data


def create_data(name):
    manager = DatasetManager(name)
    dat = manager.read_dataset()
    return dat


# split and transform data with sequence encoding mechanism


def split_prefix(dat):
    # split into training and test
    train, test = dataset_manager.split_data_strict(
        dat, train_ratio, split="temporal")

    # prefix generation of train and test data
    train_prefixes = dataset_manager.generate_prefix_data(
        train, min_prefix_length, max_prefix_length)
    test_prefixes = dataset_manager.generate_prefix_data(
        test, min_prefix_length, max_prefix_length)

    # get the label of the train and test set
    y_test = dataset_manager.get_label_numeric(test_prefixes)
    y_train = dataset_manager.get_label_numeric(train_prefixes)

    events = list(dataset_manager.get_prefix_lengths(test_prefixes))

    return train_prefixes, test_prefixes, y_train, y_test, events


def transform_data(dt_train, dt_test, y_train):
    # feature combiner and columns
    feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(
        method, **cls_encoder_args)) for method in methods])
    feature_combiner.fit(dt_train, y_train)

    # transform train dataset and add the column names back to the dataframe
    train_named = feature_combiner.transform(dt_train)
    train_named = pd.DataFrame(train_named)
    names = feature_combiner.get_feature_names()
    train_named.columns = names

    # transform test dataset
    test_named = feature_combiner.transform(dt_test)
    test_named = pd.DataFrame(test_named)
    names = feature_combiner.get_feature_names()
    test_named.columns = names

    return train_named, test_named


def attack_event_payload(dt_test, dynamic_attr):
    print('first attack generated')
    dt_test2 = dt_test.copy()
    tic = time.perf_counter()
    dt_last = dt_test2.groupby(['Case ID']).last()
    dt_last = dt_last.reset_index()
    for ii in range(0, max_prefix_length):
        for j in dynamic_attr:
            dt_last[j].loc[(dt_last.prefix_nr == ii + 1)] = random.choices(
                payload_values[j], k=len(dt_last.loc[(dt_last.prefix_nr == ii + 1)]))
    dt_test = dt_test2.set_index('Case ID', 'prefix_nr', 'event_nr')
    dt_last = dt_last.set_index('Case ID', 'prefix_nr', 'event_nr')
    dt_test.update(dt_last)
    dt_test = dt_test.reset_index()
    dt_test = dt_test.drop('level_0', axis=1)
    toc = time.perf_counter()

    print(f"First adversarial attack in {toc - tic:0.4f} seconds")
    return dt_test


def attack_all_payload(dt_test, dynamic_attr):
    print('second attack generated')
    tic = time.perf_counter()
    for j in dynamic_attr:
        dt_test[j] = random.choices(payload_values[j], k=len(dt_test))
    toc = time.perf_counter()
    print(f"Second adversarial attack in {toc - tic:0.4f} seconds")
    return dt_test


def sensivity_specifity_cutoff(y_true, y_score):

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]


def probability_to_binary(y_true, y_pred):
    cutoff = sensivity_specifity_cutoff(y_true, y_pred)
    a = np.array(y_pred).copy()
    a[a > cutoff] = 1
    a[a <= cutoff] = 0
    accuracy = accuracy_score(test_y_all, a)
    return accuracy


# PARAMETERS


params_dir = './params_dir_ML'
results_dir = './results_dir_ML_test'
column_selection = 'all'
train_ratio = 0.8
n_splits = 3
random_state = 22
n_iter = 1

# create results directory
if not os.path.exists(os.path.join(results_dir)):
    os.makedirs(os.path.join(results_dir))

encoding_dict = {
    "agg": ["static", "agg"],
    # "index": ["static", "index"]
}
encoding = []
for k, v in encoding_dict.items():
    encoding.append(k)

dataset_ref_to_datasets = {
    # "bpic2011": ["bpic2011_f%s" % formula for formula in range(2, 4)],
    "bpic2015": ["bpic2015_%s_f2" % municipality for municipality in range(5, 6)],
    #"sepsis_cases": ["sepsis_cases_2", "sepsis_cases_4"],
    #"production": ["production"],
    #"traffic_fines": ["traffic_fines_%s" % formula for formula in range(1, 2)],
    # "bpic2012": ["bpic2012_accepted", 'bpic2012_cancelled', "bpic2012_declined"],
    # "bpic2017": ["bpic2017_accepted","bpic2017_cancelled","bpic2017_refused"],
    # "hospital_billing": ["hospital_billing_%s"%suffix for suffix in [2,3]]
}
datasets = []
for k, v in dataset_ref_to_datasets.items():
    datasets.extend(v)

# classifiers dictionary
classifier_ref_to_classifiers = {
    "LRmodels": ['LR'],
    "MLmodels": ["RF"],
}
classifiers = []
for k, v in classifier_ref_to_classifiers.items():
    classifiers.extend(v)

for dataset_name in datasets:
    for cls_method in classifiers:
        for cls_encoding in encoding:
            print('Dataset:', dataset_name)
            print('Classifier', cls_method)
            print('Encoding', cls_encoding)
            dataset_manager = DatasetManager(dataset_name)
            data = create_data(dataset_name)
            method_name = "%s_%s" % (column_selection, cls_encoding)
            methods = encoding_dict[cls_encoding]

            # extract the optimal parameters
            optimal_params_filename = os.path.join(
                params_dir, "optimal_params_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name))
            if not os.path.isfile(optimal_params_filename) or os.path.getsize(optimal_params_filename) <= 0:
                print('problem')
            with open(optimal_params_filename, "rb") as fin:
                args = pickle.load(fin)
                print(args)

            cls_encoder_args = {'case_id_col': dataset_manager.case_id_col,
                                'static_cat_cols': dataset_manager.static_cat_cols,
                                'static_num_cols': dataset_manager.static_num_cols,
                                'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                                'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                                'fillna': True}

            # determine min and max (truncated) prefix lengths
            min_prefix_length = 1
            if "traffic_fines" in dataset_name:
                max_prefix_length = 10
            elif "bpic2017" in dataset_name:
                max_prefix_length = min(
                    20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
            else:
                max_prefix_length = min(
                    40, dataset_manager.get_pos_case_length_quantile(data, 0.90))

            # outfile to save the results in
            outfile = os.path.join(results_dir, "performance_results_test%s_%s_%s.csv" % (
                cls_method, dataset_name, method_name))

            # transform data
            dt_train_prefixes, dt_test_prefixes, train_y, test_y, nr_events = split_prefix(
                data)

            #dt_train_prefixes2 = dt_train_prefixes.copy()
            dt_test_prefixes2 = dt_test_prefixes.copy()
            #dt_train_prefixes3 = dt_train_prefixes.copy()
            dt_test_prefixes3 = dt_test_prefixes.copy()

            # prefix lengths
            prefix_lengths = np.zeros([max_prefix_length])
            maximal_events = dt_test_prefixes.groupby('Case ID')[['prefix_nr', 'event_nr']].max()
            maximal_events = maximal_events.reset_index()
            for i in range(0, max_prefix_length):
                if len(maximal_events[maximal_events['prefix_nr'] == i + 1]) > 0:
                    prefix_lengths[i] = len(maximal_events[maximal_events['prefix_nr'] == i + 1])
                else:
                    continue

            dynamic_cat_attributes = cls_encoder_args['dynamic_cat_cols'].copy()
            dynamic_num_attributes = cls_encoder_args['dynamic_num_cols'].copy()
            dynamic_num_attributes.remove('event_nr')
            activity_col = [
                word for word in dynamic_cat_attributes if word.startswith('Activity')]
            dynamic_cat_attributes = [
                x for x in dynamic_cat_attributes if x not in activity_col]
            dynamic_attributes = dynamic_cat_attributes + dynamic_num_attributes

            payload_values = {
                key: list(dt_train_prefixes[key].unique()) for key in dynamic_attributes}
            dt_train_prefixes.drop(columns='event_nr')
            dt_test_prefixes.drop(columns='event_nr')
            #dt_train_prefixes2.drop(columns='event_nr')
            dt_test_prefixes2.drop(columns='event_nr')
            #dt_train_prefixes3.drop(columns='event_nr')
            dt_test_prefixes3.drop(columns='event_nr')

            dt_train_named, dt_test_named = transform_data(
                dt_train_prefixes, dt_test_prefixes, train_y)

            dt_train_named_original = dt_train_named.copy()
            dt_test_named_original = dt_test_named.copy()
            scaler = MinMaxScaler()
            dt_train_named_scaled = scaler.fit_transform(dt_train_named)
            dt_test_named_scaled = scaler.transform(dt_test_named)
            dt_train_named = pd.DataFrame(
                dt_train_named_scaled, columns=dt_train_named.columns)
            dt_test_named = pd.DataFrame(
                dt_test_named_scaled, columns=dt_test_named.columns)

            # first attack
            dt_test_prefixes2 = attack_event_payload(dt_test_prefixes2, dynamic_attributes)

            dt_train_named2, dt_test_named2 = transform_data(
                dt_train_prefixes, dt_test_prefixes2, train_y)
            dt_train_named2_scaled = scaler.fit_transform(dt_train_named2)
            dt_test_named2_scaled = scaler.transform(dt_test_named2)
            dt_train_named2 = pd.DataFrame(
                dt_train_named2_scaled, columns=dt_train_named2.columns)
            dt_test_named2 = pd.DataFrame(
                dt_test_named2_scaled, columns=dt_test_named2.columns)

            # second attack
            dt_test_prefixes3 = attack_all_payload(dt_test_prefixes3, dynamic_attributes)

            dt_train_named3, dt_test_named3 = transform_data(
                dt_train_prefixes, dt_test_prefixes3, train_y)

            dt_train_named3_scaled = scaler.fit_transform(dt_train_named3)
            dt_test_named3_scaled = scaler.transform(dt_test_named3)
            dt_train_named3 = pd.DataFrame(
                dt_train_named3_scaled, columns=dt_train_named.columns)
            dt_test_named3 = pd.DataFrame(
                dt_test_named3_scaled, columns=dt_test_named.columns)

            preds_all = []
            preds_all2 = []
            preds_all3 = []

            test_y_all = []
            nr_events_all = []
            test_y_all.extend(test_y)
            nr_events_all.extend(nr_events)

            if cls_method == 'LR':

                cls1 = LogisticRegression(
                    C=2 ** args['C'], solver='saga', penalty="l1", n_jobs=-1, random_state=random_state)
                cls1.fit(dt_train_named, train_y)
                coefmodel = pd.DataFrame({'coefficients': cls1.coef_.T.tolist(), 'variable': dt_test_named.columns})
                coefficients1 = abs(np.array(coefmodel['coefficients'].apply(pd.Series).stack().reset_index(drop=True)))
                preds_pos_label_idx = np.where(cls1.classes_ == 1)[0][0]
                pred = cls1.predict_proba(dt_test_named)[
                       :, preds_pos_label_idx]
                preds_all.extend(pred)

                # first attack
                cls2 = LogisticRegression(
                    C=2 ** args['C'], solver='saga', penalty="l1", n_jobs=-1, random_state=random_state)
                cls2.fit(dt_train_named2, train_y)
                coefmodel = pd.DataFrame({'coefficients': cls2.coef_.T.tolist(), 'variable': dt_test_named2.columns})
                coefficients2 = abs(np.array(coefmodel['coefficients'].apply(pd.Series).stack().reset_index(drop=True)))
                pred2 = cls2.predict_proba(dt_test_named2)[
                        :, preds_pos_label_idx]
                preds_all2.extend(pred2)

                # second attack
                cls3 = LogisticRegression(
                    C=2 ** args['C'], solver='saga', penalty="l1", n_jobs=-1, random_state=random_state)
                cls3.fit(dt_train_named3, train_y)
                coefmodel = pd.DataFrame({'coefficients': cls3.coef_.T.tolist(), 'variable': dt_test_named3.columns})
                coefficients3 = abs(np.array(coefmodel['coefficients'].apply(pd.Series).stack().reset_index(drop=True)))
                pred3 = cls3.predict_proba(dt_test_named3)[
                        :, preds_pos_label_idx]
                preds_all3.extend(pred3)

            elif cls_method == "RF":
                cls1 = RandomForestClassifier(n_estimators=args['max_features'],
                                              n_jobs=-1,
                                              random_state=random_state)
                cls1.fit(dt_train_named, train_y)

                cls2 = RandomForestClassifier(n_estimators=args['max_features'],
                                              n_jobs=-1,
                                              random_state=random_state)
                cls2.fit(dt_train_named2, train_y)

                cls3 = RandomForestClassifier(n_estimators=args['max_features'],
                                              n_jobs=-1,
                                              random_state=random_state)
                cls3.fit(dt_train_named3, train_y)

                # predictions
                preds_pos_label_idx = np.where(cls1.classes_ == 1)[0][0]
                pred = cls1.predict_proba(dt_test_named)[:, preds_pos_label_idx]
                preds_all.extend(pred)

                pred2 = cls2.predict_proba(dt_test_named2)[:, preds_pos_label_idx]
                preds_all2.extend(pred2)

                pred3 = cls3.predict_proba(dt_test_named3)[:, preds_pos_label_idx]
                preds_all3.extend(pred3)

            # before attacking
            auc_total1 = roc_auc_score(test_y_all, preds_all)
            print(auc_total1)
            accuracy1 = probability_to_binary(test_y_all, preds_all)
            print('accuracy', accuracy1)

            # first attack
            print('auc2', roc_auc_score(test_y_all, preds_all2))
            accuracy2 = probability_to_binary(test_y_all, preds_all2)
            print('accuracy2', accuracy2)

            # second attack
            print('auc3', roc_auc_score(test_y_all, preds_all3))
            accuracy3 = probability_to_binary(test_y_all, preds_all3)
            print('accuracy3', accuracy3)

            with open(outfile, 'w') as fout:
                fout.write("%s;%s;%s;%s;%s;%s\n" % ("dataset", "method", "cls", "nr_events", "metric", "score"))

                dt_results = pd.DataFrame({"actual": test_y_all, "predicted": preds_all, "nr_events": nr_events_all})
                for nr_events, group in dt_results.groupby("nr_events"):
                    if len(set(group.actual)) < 2:
                        fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method,
                                                               nr_events, -1, "auc", np.nan))
                    else:
                        fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, nr_events, -1,
                                                               "auc", roc_auc_score(group.actual, group.predicted)))
                fout.write("%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, "auc",
                                                    roc_auc_score(dt_results.actual, dt_results.predicted)))

                # attack1
                dt_results = pd.DataFrame({"actual": test_y_all, "predicted": preds_all2, "nr_events": nr_events_all})
                for nr_events, group in dt_results.groupby("nr_events"):
                    if len(set(group.actual)) < 2:
                        fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method,
                                                               nr_events, -1, "auc", np.nan))
                    else:
                        fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, nr_events, -1,
                                                               "auc", roc_auc_score(group.actual, group.predicted)))
                fout.write("%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, "auc",
                                                    roc_auc_score(dt_results.actual, dt_results.predicted)))

                # attack2
                dt_results = pd.DataFrame({"actual": test_y_all, "predicted": preds_all3, "nr_events": nr_events_all})
                for nr_events, group in dt_results.groupby("nr_events"):
                    if len(set(group.actual)) < 2:
                        fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, nr_events,
                                                               -1, "auc", np.nan))
                    else:
                        fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, nr_events,
                                                               -1, "auc", roc_auc_score(group.actual, group.predicted)))
                fout.write("%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, "auc",
                                                    roc_auc_score(dt_results.actual, dt_results.predicted)))
