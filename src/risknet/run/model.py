'''This .py file defines training hyperparameters and returning evaluation metrics.'''

#Global imports
import time
from sklearn import metrics
from pandas import DataFrame
import xgboost as xgb
from bayes_opt import BayesianOptimization
import logging
from xgboost.core import Booster

import pandas as pd
from typing import List, Dict, Tuple
import pickle

from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score, auc, average_precision_score, mean_absolute_error
import matplotlib.pyplot as plt

#Variables
logger = logging.getLogger("freelunch")
cat_label: str = "default"

#XGB Instance:
class XGBCVTrain(object):

    def __init__(self):
        self.lr = None
        self.train_auc = None
        self.importance = None
        self.bst: Booster = Booster()

    def train(self, train_data: DataFrame, train_label: DataFrame, val_data: DataFrame, val_label: DataFrame, target: str, objective: str, maximize: bool = True,
              min_boosting_rounds: int = 3, max_boosting_rounds: int = 100, bayes_n_init: int =  50, bayes_n_iter: int = 5):

        """Run xgboost training job.  Consider the following evaluation metrics and be mindful of whether you are maximizing or minimizing:
        # error will require that maximize equals false.. we want to minimize..
        #error
        #error@t: a different than 0.5 binary classification threshold value could be specified by providing a numerical value through ‘t’.
        #auc: Area under the curve
        # aucpr: Area under the PR curve
        #map: Mean Average Precision
        #rmse: root mean square error (regression)
        # mae: mean absolute error (regression)
        # mlogloss: Multiclass logloss.  This is more like how you understand logloss.

                """

        time_begin = time.time()

        logger.info("BEGIN LOGIT SMALL TRAINING JOB")

        logger.info("Train on full data")

        logger.info("Convert to DMatrix")

        dtrain = xgb.DMatrix(train_data.values, label=train_label.values,
                             feature_names=train_data.columns.values.tolist())
        
        dval = xgb.DMatrix(val_data, label=val_label, nthread=-1)

        logger.info("Define Evaluation Function")

        def xgb_evaluate(eta, gamma, max_depth, min_child_weight, subsample, colsample_bytree, l_2, l_1, rounds):
            param = {'eta': eta,
                     'gamma': gamma,
                     'max_depth': int(max_depth),
                     'min_child_weight': int(min_child_weight),
                     'subsample': subsample,
                     'colsample_bytree': colsample_bytree,
                     'lambda': l_2,
                     'alpha': l_1,
                     #'silent': 1, #The code in my notebook says "silent" is not used (removed by #EC)
                     'objective': 'binary:logistic',
                     'booster': 'gbtree',
                     'verbosity': 2,
                     }

            test_metric: float = \
            xgb.cv(param, dtrain, num_boost_round=int(rounds), nfold=5, stratified=True, early_stopping_rounds=20,
                   metrics=objective,
                   maximize=maximize)['test-' + objective[0] + '-mean'].mean()

            if maximize:
                return test_metric
            else:
                return -test_metric

        bounds = {'eta': (0.01, 0.1),
                  'gamma': (0.05, 1.0),
                  'max_depth': (3, 25),
                  'min_child_weight': (3, 7),
                  'subsample': (0.6, 1.0),
                  'colsample_bytree': (0.6, 1.0),
                  'l_2': (0.01, 1.0),
                  'l_1': (0, 1.0),
                  'rounds': (min_boosting_rounds, max_boosting_rounds)}

        xgb_bo = BayesianOptimization(xgb_evaluate, pbounds=bounds)

        #xgb_bo.set_gp_param(kappa=2.576) #Removed by EC because it didn't like the way I passed in kappa below and wouldn't accept set_gp_param
        xgb_bo.maximize(init_points=bayes_n_init, n_iter=bayes_n_iter)


        logger.info("Best parameters are:")

        logger.info(str(xgb_bo.max['params']))

        # todo print/log ideal params here

        param = {'eta': xgb_bo.max['params']['eta'],
                 'gamma': xgb_bo.max['params']['gamma'],
                 'max_depth': int(xgb_bo.max['params']['max_depth']),
                 'min_child_weight': int(xgb_bo.max['params']['min_child_weight']),
                 'subsample': xgb_bo.max['params']['subsample'],
                 'colsample_bytree': xgb_bo.max['params']['colsample_bytree'],
                 'lambda': xgb_bo.max['params']['l_2'],
                 'alpha': xgb_bo.max['params']['l_1'],
                 #'silent: 1, ' #Commented out by EC
                 'objective': 'binary:logistic',
                 'eval_metric': "auc",
                 'booster': 'gbtree',
                 'verbosity': 2}

        self.bst = xgb.train(param, dtrain, num_boost_round=int(xgb_bo.max['params']['rounds']), evals =[(dval, 'val')],
                             maximize=maximize, verbose_eval=True) #.train returns a Booster object


        logger.info("Importance: ")

        logger.info(
            sorted(self.bst.get_score(importance_type='gain'), key=self.bst.get_score(importance_type='gain').get,
                   reverse=True))

        self.importance = sorted(self.bst.get_score(importance_type='gain'),
                                 key=self.bst.get_score(importance_type='gain').get, reverse=True)

        logger.info("run predictions on train and  datasets")

        train_label['prediction'] = self.bst.predict(dtrain)
        #EC: apparently this version of XGB says that self.bst doesn't have an attribute best_ntree_limit...will remove
        #ntree_limit=self.bst.best_iteration

        logger.info("Generate Metrics")

        fpr, tpr, _ = metrics.roc_curve(train_label[target], train_label['prediction'], pos_label=1)

        self.train_auc: str = str(metrics.auc(fpr, tpr))
        logger.info("Train AUC: " + self.train_auc)

        logger.info("XGB TRAINING AND EVALUATION OUTPUT DONE")

        time_end = time.time()

        total = time_end - time_begin

        logger.info("Total time: " + str(round((total / 60), 2)) + "minutes")

    def predict(self, scoring_data: DataFrame):

        dscore = xgb.DMatrix(scoring_data.values, feature_names=scoring_data.columns.values.tolist())
        return self.bst.predict(dscore)
        #EC note: again, remove ntree_limit=self.bst.best_iteration. Instead using `iteration_range`
        #EC note 2: removeed iteration_range because it tanked performance :/

    def get_auc(self):
        return self.train_auc

    def get_importance(self):
        return self.importance
    


#Training a Model
def xgb_train(fm_root, baseline=False, cat_label='default'):
    #Set up, initialize:
    #Create XGB object
    xgb_cv: XGBCVTrain = XGBCVTrain()

    #Set up DF
    df = pd.read_pickle(fm_root + 'df.pkl') #pull scaled df and labels

    non_train_columns: List[str] = ['default', 'undefaulted_progress', 'flag', 'loan_sequence_number'] #Add loan_seq_num EC
    
    if baseline == True:
        train_columns : List[str] = ['credit_score']
    else:
        train_columns: List[str] = [i for i in df.columns.to_list() if i not in non_train_columns]

    #Get minmax values
    df_train_minmax: DataFrame = df.loc[df['flag'] == 'train'].loc[:, train_columns]
    df_train_label: DataFrame = df.loc[df['flag'] == 'train'].loc[:, ['default']]
    df_train_label_reg: DataFrame = df.loc[df['flag'] == 'train'].loc[:, ['undefaulted_progress']]
    df_test_minmax: DataFrame = df.loc[df['flag'] == 'test'].loc[:, train_columns]
    df_test_label: DataFrame = df.loc[df['flag'] == 'test'].loc[:, ['default']]
    df_test_label_reg: DataFrame = df.loc[df['flag'] == 'test'].loc[:, ['undefaulted_progress']]
    df_val_minmax: DataFrame = df.loc[df['flag'] == 'val'].loc[:, train_columns]
    df_val_label: DataFrame = df.loc[df['flag'] == 'val'].loc[:, ['default']]
    df_val_label_reg: DataFrame = df.loc[df['flag'] == 'val'].loc[:, ['undefaulted_progress']]

    #Train model instance
    xgb_cv.train(df_train_minmax, df_train_label, df_val_minmax, df_val_label, cat_label, ['auc'], True, 3, 100, 50, 5)

    # predict on training set using XGB_CV
    df_train_label['xgb_score'] = xgb_cv.predict(df_train_minmax)
    df_test_label['xgb_score'] = xgb_cv.predict(df_test_minmax)
    df_val_label['xgb_score'] = xgb_cv.predict(df_val_minmax)

    with open(fm_root + 'xgb_cv.pkl', 'wb') as f:
        pickle.dump(xgb_cv, f)

    return [df_train_label, df_test_label, df_val_label]

def xgb_eval(data):
    return [xgb_auc(data), xgb_pr(data), xgb_recall(data)]

def xgb_auc(data):
    df_train_label, df_test_label, df_val_label = data
    
    #training AUC
    fpr, tpr, thresholds = roc_curve(df_train_label[cat_label], df_train_label['xgb_score'], pos_label=1)
    xgb_train_auc: float = auc(fpr, tpr)

    #testing AUC
    fpr, tpr, thresholds = roc_curve(df_test_label[cat_label], df_test_label['xgb_score'], pos_label=1)
    xgb_test_auc: float = auc(fpr, tpr)

    #validation AUC
    fpr, tpr, thresholds = roc_curve(df_val_label[cat_label], df_val_label['xgb_score'], pos_label=1)
    xgb_val_auc: float = auc(fpr, tpr)

    aucs = [xgb_train_auc, xgb_test_auc, xgb_val_auc]
    return aucs

def xgb_pr(data):
    '''Precision'''
    df_train_label, df_test_label, df_val_label = data

    #Train, test, validation precision
    xgb_train_av_pr: float = average_precision_score(df_train_label[cat_label], df_train_label['xgb_score'], pos_label=1)
    xgb_test_av_pr: float = average_precision_score(df_test_label[cat_label], df_test_label['xgb_score'], pos_label=1)
    xgb_val_av_pr: float = average_precision_score(df_val_label[cat_label], df_val_label['xgb_score'], pos_label=1)

    av_pr: List[float] = [xgb_train_av_pr, xgb_test_av_pr, xgb_val_av_pr]
    return av_pr

def xgb_recall(data):
    '''Recall'''
    df_train_label, df_test_label, df_val_label = data
    #xgb_train_precision, xgb_train_recall, xgb_train_thresholds = precision_recall_curve(df_train_label[cat_label], df_train_label['xgb_score'])
    #xgb_test_precision, xgb_test_recall, xgb_test_thresholds = precision_recall_curve(df_test_label[cat_label], df_test_label['xgb_score'])
    xgb_val_precision, xgb_val_recall, xgb_val_thresholds = precision_recall_curve(df_val_label[cat_label], df_val_label['xgb_score'])

    val_recall: List[float] = [xgb_val_precision, xgb_val_recall, xgb_val_thresholds]
    return val_recall
