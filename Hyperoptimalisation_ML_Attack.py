# import packages
import pandas as pd
import numpy as np
import os
import EncoderFactory
from DatasetManager import DatasetManager
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle
import hyperopt
from hyperopt import hp, Trials, fmin, tpe, STATUS_OK
from sklearn.preprocessing import MinMaxScaler

# parameters
params_dir = './params_dir_ML'
column_selection = 'all'
train_ratio = 0.8
n_splits = 3
random_state = 22
n_iter = 1

# create results directory
if not os.path.exists(os.path.join(params_dir)):
    os.makedirs(os.path.join(params_dir))

encoding_dict = {
    "agg": ["static", "agg"],
    # "index": ["static", "index"]
}
encoding = []
for k, v in encoding_dict.items():
    encoding.append(k)
dataset_ref_to_datasets = {
    #"bpic2011": ["bpic2011_f%s"%formula for formula in range(2,4)],
    #"bpic2015": ["bpic2015_%s_f2"%(municipality) for municipality in range(5,6)],
    "sepsis_cases": ["sepsis_cases_2"]
       # ,"sepsis_cases_4"],
    #"production": ["production"],
    # "bpic2012": ["bpic2012_accepted,"bpic2012_cancelled","bpic2012_declined"],
    # "bpic2017": ["bpic2017_accepted","bpic2017_cancelled","bpic2017_refused"],
    # "hospital_billing": ["hospital_billing_%s"%suffix for suffix in [2,3]],
    #"traffic_fines": ["traffic_fines_%s" % formula for formula in range(1, 2)],
}

datasets = []
for k, v in dataset_ref_to_datasets.items():
    datasets.extend(v)

# classifiers dictionary
classifier_ref_to_classifiers = {
    "LRmodels": ["LR"],
   # "MLmodels": ["RF"],
}
classifiers = []
for k, v in classifier_ref_to_classifiers.items():
    classifiers.extend(v)
    
# create and evaluate function


def create_and_evaluate_model(args): 
    global trial_nr
    trial_nr += 1
    score = 0
    for cv_iter in range(n_splits):
        dt_test_prefixes = dt_prefixes[cv_iter]
        dt_train_prefixes = pd.DataFrame()
        for cv_train_iter in range(n_splits): 
            if cv_train_iter != cv_iter:
                dt_train_prefixes = pd.concat([dt_train_prefixes, dt_prefixes[cv_train_iter]], axis=0)

        preds_all = []
        test_y_all = []
        test_y = dataset_manager.get_label_numeric(dt_test_prefixes)
        train_y = dataset_manager.get_label_numeric(dt_train_prefixes)  
        test_y_all.extend(test_y) 
        dt_train_prefixes.drop(columns='event_nr')
        dt_test_prefixes.drop(columns='event_nr')
       
        # feature combiner
        feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args))
                                         for method in methods])
        feature_combiner.fit(dt_train_prefixes, train_y)
        # transform train dataset
        dt_train_named = feature_combiner.transform(dt_train_prefixes)
        dt_train_named = pd.DataFrame(dt_train_named)
        names = feature_combiner.get_feature_names()
        dt_train_named.columns = names
        # dt_train_named = dt_train_named.drop('static__label_deviant', 1)
        # dt_train_named = dt_train_named.drop('static__label_regular', 1)

        # transform test dataset
        dt_test_named = feature_combiner.transform(dt_test_prefixes)
        dt_test_named = pd.DataFrame(dt_test_named)
        names = feature_combiner.get_feature_names()
        dt_test_named.columns = names
        # dt_test_named = dt_test_named.drop('static__label_deviant', 1)
        # dt_test_named = dt_test_named.drop('static__label_regular', 1)
        
        scaler = MinMaxScaler()
        scaler.fit(dt_train_named)
        dt_train_named2 = scaler.transform(dt_train_named)  
        dt_test_named2 = scaler.transform(dt_test_named)
        dt_train_named = pd.DataFrame(dt_train_named2, columns=dt_train_named.columns)
        dt_test_named = pd.DataFrame(dt_test_named2, columns=dt_test_named.columns)
        cls = None
        if cls_method == "LR":
            cls = LogisticRegression(C=2**args['C'], solver='saga', penalty="l1", n_jobs=-1, random_state=random_state)
            cls.fit(dt_train_named, train_y)  # apply scaling on training data
            preds_pos_label_idx = np.where(cls.classes_ == 1)[0][0]
            pred = cls.predict_proba(dt_test_named)[:, preds_pos_label_idx]
            preds_all.extend(pred)
        elif cls_method == "RF":
            cls = RandomForestClassifier(max_features=args['max_features'],
                                         n_jobs=-1,
                                         random_state=random_state)
            cls.fit(dt_train_named, train_y)
            preds_pos_label_idx = np.where(cls.classes_ == 1)[0][0]
            pred = cls.predict_proba(dt_test_named)[:, preds_pos_label_idx]
            preds_all.extend(pred)
      
        score += roc_auc_score(test_y_all, preds_all)
        for k, v in args.items():
            fout_all.write("%s;%s;%s;%s;%s;%s;%s\n" % (trial_nr, dataset_name,
                                                       cls_method, method_name, k, v, score / n_splits))
        fout_all.write("%s;%s;%s;%s;%s\n" % (trial_nr, dataset_name, cls_method, method_name, 0))   
    fout_all.flush()
    return {'loss': -score / n_splits, 'status': STATUS_OK, 'model': cls}


# print dataset name
for dataset_name in datasets:
    # read the data
    dataset_manager = DatasetManager(dataset_name)
    data = dataset_manager.read_dataset()

    for cls_method in classifiers:
        for cls_encoding in encoding:
            print('Dataset:', dataset_name)
            print('Classifier', cls_method)
            print('Encoding', cls_encoding)
            
            method_name = "%s_%s" % (column_selection, cls_encoding)
            methods = encoding_dict[cls_encoding]
            cls_encoder_args = {'case_id_col': dataset_manager.case_id_col, 
                                'static_cat_cols': dataset_manager.static_cat_cols,
                                'static_num_cols': dataset_manager.static_num_cols,
                                'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                                'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                                'fillna': True,
                                'max_events': None}

            # determine min and max (truncated) prefix lengths
            min_prefix_length = 1
            if "traffic_fines" in dataset_name:
                max_prefix_length = 10
            elif "bpic2017" in dataset_name:
                max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
            else:
                max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))
            
            # split into training and test
            train, _ = dataset_manager.split_data_strict(data, train_ratio, split="temporal")

            # prepare chunks for CV
            dt_prefixes = []
            class_ratios = []
            for train_chunk, test_chunk in dataset_manager.get_stratified_split_generator(train, n_splits=n_splits):
                class_ratios.append(dataset_manager.get_class_ratio(train_chunk))
                # generate data where each prefix is a separate instance
                dt_prefixes.append(dataset_manager.generate_prefix_data(test_chunk,
                                                                        min_prefix_length, max_prefix_length))
            del train
        
            # set up search space
            space = {}
            if cls_method == "RF":
                space = {'max_features': hp.uniform('max_features', 0, 1)}
    
            if cls_method == "LR":
                space = {'C': hp.uniform('C', -15, 15)}
            
            # optimize parameters
            trial_nr = 0
            trials = Trials()
            fout_all = open(os.path.join(params_dir, "param_optim_all_trials_%s_%s_%s.csv" % (cls_method, dataset_name,
                                                                                              method_name)), "w")
            if "prefix" in method_name:
                fout_all.write("%s;%s;%s;%s;%s;%s;%s;%s\n" % ("iter", "dataset", "cls", "method", "nr_events", "param",
                                                              "value", "score"))
            else:
                fout_all.write("%s;%s;%s;%s;%s;%s;%s\n" % ("iter", "dataset", "cls", "method", "param", "value",
                                                           "score"))
            best = fmin(create_and_evaluate_model, space, algo=tpe.suggest, max_evals=4, trials=trials)
            fout_all.close()
        
            # write the best parameters
            best_params = hyperopt.space_eval(space, best)
            outfile = os.path.join(params_dir, "optimal_params_%s_%s_%s.pickle" % (cls_method, dataset_name,
                                                                                   method_name))
            # write to file
            with open(outfile, "wb") as fout:
                pickle.dump(best_params, fout)
