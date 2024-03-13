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
    '''
    Generates an XGB object that undergoes hyperparameter tuning

    Parameters
    ----------
    train_data: DataFrame
        DataFrame that contains all the training data. Often called X_train
    train_label: DataFrame
        DataFrame that contains the training labels. Often called y_train
    val_data: DataFrame
        Dataframe that contains the validation data. Often called X_val
    val_label: DataFrame
        Dataframe that contains the validation labels. Often called y_val
    target: str
        String that defines the y variable. Often defined as "default"
    objective: List[str]
        String that describes the objective(s) to maximize. In our case, often called ['auc']
    maximize: Boolean
        String that describes whether to maximize (True) or minimize (False) the objective.
        Default value is True.
    min_boosting_rounds: Int.
        Integer that describes the minimum number of bounds that a training model can perform. This is fed into hyperparameter tuning.
        Default value is 3.
    max_boosting_rounds: Int
        Integer that describes the maximum number of bounds that a training model can perform. This is fed into hyperparameter tuning.
        Default value is 100.
    bayes_n_init: Int.
        Integer that defines the number of random starting points to start iterating for hyperparams. This is fed into hyperparameter tuning.
        Default values is 50.
    bayes_n_iter: Int.
        Integer that defines the number of steps of bayesian optimization to perform. This is fed into hyperparameter tuning.
    eta: Float
        Float value between [0, 1] that defines the learning rate.
        In our model, this value is optimized between (0.01, 0.1).
    gamma: Float
        Float value that regularizes the minimum reduction in loss to justify a partition in a leaf node.
        Basically prevents the algorithm from splitting if it doesn't contribute a significant improvement in performance.
        In our model, this value is optimized between (0.05, 1.0)
    max_depth: Int
        Integer value that controls the maximum depth of each tree in the boosting process.
        In our model, this value is optimized between (3, 25)
    min_child_weight: Int
        Integer value that regularizes the minimum number of instances required to create a new node.
        Like gamma, prevents the algorithm from creating new nodes if it doesn't significantly help performance.
        In our model, this value is optimized between (3, 7)
    subsample: Float
        Float value that defines the fraction of samples used to fit base learners. 
        Introduces randomness in the training process.
        If the value is < 1.0, we are subsampling.
        In our model, this value is optimized between (0.6, 1.0)
    colsample_by_tree: Float
        Float value that defines the fraction of features used to fit base learners.
        In our model, this value is optimized between (0.6, 1.0)
    l_2: Float
        Float value that is our L2 regularization term. Also called lambda.
        Penalizes large weights to prevent overfitting.
        In our model, this value is optimized between (0.01, 1.0)
    l_1: Float
        Float value that is our L1 regularization term. Also called alpha.
        Penalizes large coefficients but uses the L1 norm instead of the L2 norm.
        In our model, this value is optimized between (0.1, 1.0)

    Attributes
    ----------
    self.lr: Float
        Defines the learning rate of the model
        Default value is None
    self.train_auc: Float
        Stores value of AUC as calculated using sklearn.metric
        Default value is None
    self.importance: List[str]
        Returns a list of features that are sorted by AUC gain (as calculated by the XGB object)
        Default value is None
    self.bst: Booster object
        Stores the trained XGBoost object trained on ideal hyperparameters and data
        Default value is a default Booster() object
    self.time = None
        Stores the time taken to train the XGB object.
    Methods
    -------
    train(self, train_data: DataFrame, train_label: DataFrame, val_data: DataFrame, val_label: DataFrame, target: str, objective: str, maximize: bool = True,
              min_boosting_rounds: int = 3, max_boosting_rounds: int = 100, bayes_n_init: int =  50, bayes_n_iter: int = 5)
    Function that trains a booster object, runs through hyperparameter tuning, and then feeds hyperparameters into a XGB object to predict/evaluate

    xgb_evaluate(eta, gamma, max_depth, min_child_weight, subsample, colsample_bytree, l_2, l_1, rounds)
    Function that defines how to run through hyperparameter tuning.

    predict(self, scoring_data: DataFrame)
    Function that uses the booster object to predict labels
    '''
    def __init__(self):
        self.lr = None
        self.train_auc = None
        self.importance = None
        self.bst: Booster = Booster()
        self.time = None

    def train(self, train_data: DataFrame, train_label: DataFrame, val_data: DataFrame, val_label: DataFrame, target: str, objective: str, maximize: bool = True,
              min_boosting_rounds: int = 3, max_boosting_rounds: int = 100, bayes_n_init: int =  50, bayes_n_iter: int = 5):
        '''
        Function that trains a booster object, runs through hyperparameter tuning, and then feeds hyperparameters into a XGB object to predict/evaluate

        Parameters
        ----------
        train_data: DataFrame
            DataFrame that contains all the training data. Often called X_train
        train_label: DataFrame
            DataFrame that contains the training labels. Often called y_train
        val_data: DataFrame
            Dataframe that contains the validation data. Often called X_val
        val_label: DataFrame
            Dataframe that contains the validation labels. Often called y_val
        target: str
            String that defines the y variable. Often defined as "default"
        objective: List[str]
            String that describes the objective(s) to maximize. In our case, often called ['auc']
        maximize: Boolean
            String that describes whether to maximize (True) or minimize (False) the objective.
            Default value is True.
        min_boosting_rounds: Int.
            Integer that describes the minimum number of bounds that a training model can perform. This is fed into hyperparameter tuning.
            Default value is 3.
        max_boosting_rounds: Int
            Integer that describes the maximum number of bounds that a training model can perform. This is fed into hyperparameter tuning.
            Default value is 100.
        bayes_n_init: Int.
            Integer that defines the number of random starting points to start iterating for hyperparams. This is fed into hyperparameter tuning.
            Default values is 50.
        bayes_n_iter: Int.
            Integer that defines the number of steps of bayesian optimization to perform. This is fed into hyperparameter tuning.
        scoring_data: DataFrame
            DataFrame that contains the data to predict with

        Returns
        -------
        None

        Note
        -------
        Assumes the training dataset has been cleaned and scaled using the processing files. 
        Also assumes that the y training and testing labels have both 0 and 1 values.
        '''

        time_begin = time.time()

        logger.info("BEGIN LOGIT SMALL TRAINING JOB")

        logger.info("Train on full data")

        logger.info("Convert to DMatrix")

        dtrain = xgb.DMatrix(train_data.values, label=train_label.values,
                             feature_names=train_data.columns.values.tolist())
        
        dval = xgb.DMatrix(val_data, label=val_label, nthread=-1)

        logger.info("Define Evaluation Function")

        def xgb_evaluate(eta, gamma, max_depth, min_child_weight, subsample, colsample_bytree, l_2, l_1, rounds):
            '''
            Defines how to use different hyperparameters and evaluate the effect on the model's performance.

            Parameters
            -------
            eta: Float
                Float value between [0, 1] that defines the learning rate.
                In our model, this value is optimized between (0.01, 0.1).
            gamma: Float
                Float value that regularizes the minimum reduction in loss to justify a partition in a leaf node.
                Basically prevents the algorithm from splitting if it doesn't contribute a significant improvement in performance.
                In our model, this value is optimized between (0.05, 1.0)
            max_depth: Int
                Integer value that controls the maximum depth of each tree in the boosting process.
                In our model, this value is optimized between (3, 25)
            min_child_weight: Int
                Integer value that regularizes the minimum number of instances required to create a new node.
                Like gamma, prevents the algorithm from creating new nodes if it doesn't significantly help performance.
                In our model, this value is optimized between (3, 7)
            subsample: Float
                Float value that defines the fraction of samples used to fit base learners. 
                Introduces randomness in the training process.
                If the value is < 1.0, we are subsampling.
                In our model, this value is optimized between (0.6, 1.0)
            colsample_by_tree: Float
                Float value that defines the fraction of features used to fit base learners.
                In our model, this value is optimized between (0.6, 1.0)
            l_2: Float
                Float value that is our L2 regularization term. Also called lambda.
                Penalizes large weights to prevent overfitting.
                In our model, this value is optimized between (0.01, 1.0)
            l_1: Float
                Float value that is our L1 regularization term. Also called alpha.
                Penalizes large coefficients but uses the L1 norm instead of the L2 norm.
                In our model, this value is optimized between (0.1, 1.0)

            Returns
            -------
            Integer, either test_metric or -test_metric.
            This output will be used when optimizing hyperparameter tuning using bayesian optimization.
            '''
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
        #In this case, n_init = # of random starting points to start iterating for hyperparams
        #In this case, n_iter = # of steps of bayesian optimization to perform
        xgb_bo.maximize(init_points=bayes_n_init, n_iter=bayes_n_iter)
        #This is what generates iter | target | x | etc.

        #These are the best parameters according to logger
        logger.info("Best/max parameters are:")
        logger.info(str(xgb_bo.max['params'])) # todo print/log ideal params here

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
        
        logger.info("Return numerical score for each feature of how it improves performance")
        logger.info(self.bst.get_score(importance_type='gain'))

        logger.info("run predictions on train and  datasets")

        train_label['prediction'] = self.bst.predict(dtrain)
        #EC: apparently this version of XGB says that self.bst doesn't have an attribute best_ntree_limit...will remove
        #ntree_limit=self.bst.best_iteration

        fpr, tpr, _ = metrics.roc_curve(train_label[target], train_label['prediction'], pos_label=1)

        self.train_auc: str = str(metrics.auc(fpr, tpr))
        logger.info("Train AUC: " + self.train_auc)

        logger.info("XGB TRAINING AND EVALUATION OUTPUT DONE")

        time_end = time.time()

        total = time_end - time_begin

        logger.info("Total time: " + str(round((total / 60), 2)) + "minutes")

        self.time = float(round(total / 60, 2))

    def predict(self, scoring_data: DataFrame):
        '''
        Function that uses the booster object to predict labels
        
        Parameters
        -------
        scoring_data: pd.DataFrame

        Returns
        -------
        pd.DataFrame containing one column of 0/1 True/False (predicting mortage default)

        Notes
        -------
        Assumes scoring data has the same column names/features as the training data
        '''

        dscore = xgb.DMatrix(scoring_data.values, feature_names=scoring_data.columns.values.tolist())
        return self.bst.predict(dscore)
        #EC note: again, remove ntree_limit=self.bst.best_iteration. Instead using `iteration_range`
        #EC note 2: removed iteration_range because it tanked performance :/

    def get_auc(self):
        '''
        Function gets AUC
        
        Parameters: None

        Returns
        -------
        Training AUC as defined by self.train_auc
        '''
        return self.train_auc

    def get_importance(self):
        '''
        Function gets importance of features
        
        Parameters: None

        Returns
        -------
        Importance of features as stored in self.importance

        Notes
        -------
        Assumes xgb_evaluate has been run; otherwise will return None.
        '''
        return self.importance
    
    def get_time(self):
        '''
        Function returns time to train model
        
        Parameters: None

        Returns
        -------
        Time to train as stored in self.time
        '''
        return self.time
    


#Training a Model
def xgb_train(df, fm_root, baseline=False, cat_label='default'):
    '''
    Function run in pipeline.py that actually uses real data to initialize/run a model.
    Heavily relies on XGBCV() class as defined above.

    Parameters
    -------
    df: pd.DataFrame
        Cleaned, scaled Dataframe as passed in from previous step in pipeline

    baseline: Boolean, optional
        Determines whether or not to use only credit score as a feature (True) or all features (False)
        Default value is False

    cat_label: String, optional
        Defines the name of the y-label. 
        Default value is 'default'
        
    Returns
    -------
    List[List[List[Float], List[Float], List[Float]], Float]
    Returns predicted labels + time in following format: [[df_train_label, df_val_label, df_test_label], xgb_cv.get_time()]

    Notes
    -------
    Assumes that all previous steps in pipeline.py have been run before passing in cleaned, processed dataset
    '''
    #Set up, initialize:
    #Create XGB object
    xgb_cv: XGBCVTrain = XGBCVTrain()

    #Set up DF
    #df = pd.read_pickle(fm_root + 'df.pkl') #pull scaled df and labels
    df = df.merge(pd.read_pickle(fm_root + 'df.pkl')) #merge FE df + the clean df from df.pkl

    #print(df.info(verbose=True)) #Prints all columns of df

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

    return [[df_train_label, df_val_label, df_test_label], xgb_cv.get_time()]

def xgb_eval(data):
    '''
    Calculates AUC, precision, and recall given labels and true values

    Parameters
    -------
    data: List[List[List[Float], List[Float], List[Float]], Float]
        Uses output of xgb_train. That is, [[df_train_label, df_val_label, df_test_label], xgb_cv.get_time()]
    
    Returns
    -------
    List[List[Float], List[Float], List[Float]]
    For each metric, return a list of [train_results, val_results, test_results].
    Will be in the order of AUC, precision, and then recall.
    '''
    return [xgb_auc(data), xgb_pr(data), xgb_recall(data)]


def xgb_auc(data):
    '''
    Gets the training, validation, and testing AUC given training, validation, and testing predicted labels.

    Parameters
    ----------
    data: List[List[List[Float], List[Float], List[Float]], Float]
        Uses output of xgb_train. That is, [[df_train_label, df_val_label, df_test_label], xgb_cv.get_time()]
    
    Returns
    -------
    List[Float]
        A list of three AUC values corresponding to training, validation, and test sets.
    
    Note
    -------
    Uses sklearn.metrics.auc to calculate AUC
    '''
    df_train_label, df_val_label, df_test_label = data
    
    #training AUC
    fpr, tpr, thresholds = roc_curve(df_train_label[cat_label], df_train_label['xgb_score'], pos_label=1)
    xgb_train_auc: float = auc(fpr, tpr)

    #validation AUC
    fpr, tpr, thresholds = roc_curve(df_val_label[cat_label], df_val_label['xgb_score'], pos_label=1)
    xgb_val_auc: float = auc(fpr, tpr)

    #testing AUC
    fpr, tpr, thresholds = roc_curve(df_test_label[cat_label], df_test_label['xgb_score'], pos_label=1)
    xgb_test_auc: float = auc(fpr, tpr)

    aucs = [xgb_train_auc, xgb_val_auc, xgb_test_auc]
    return aucs


def xgb_pr(data):
    '''
    Gets the training, validation, and testing precision given training, validation, and testing predicted labels.

    Parameters
    ----------
    data: List[List[List[Float], List[Float], List[Float]], Float]
        Uses output of xgb_train. That is, [[df_train_label, df_val_label, df_test_label], xgb_cv.get_time()]
    
    Returns
    -------
    List[Float]
        A list of three precision values corresponding to training, validation, and test sets.

    Note
    -------
    Uses sklearn.metrics.average_precision_score to calculate precision
    '''
    df_train_label, df_val_label, df_test_label = data

    #Train, test, validation precision
    xgb_train_av_pr: float = average_precision_score(df_train_label[cat_label], df_train_label['xgb_score'], pos_label=1)
    xgb_test_av_pr: float = average_precision_score(df_test_label[cat_label], df_test_label['xgb_score'], pos_label=1)
    xgb_val_av_pr: float = average_precision_score(df_val_label[cat_label], df_val_label['xgb_score'], pos_label=1)

    av_pr: List[float] = [xgb_train_av_pr, xgb_val_av_pr, xgb_test_av_pr]
    return av_pr

def xgb_recall(data):
    '''
    Gets the training, validation, and testing recall given training, validation, and testing predicted labels.

    Parameters
    ----------
    data: List[List[List[Float], List[Float], List[Float]], Float]
        Uses output of xgb_train. That is, [[df_train_label, df_val_label, df_test_label], xgb_cv.get_time()]
    
    Returns
    -------
    List[Float]
        A list of three recall values corresponding to training, validation, and test sets.

    Note
    -------
    Uses sklearn.metrics.precision_recall_curve to calculate recall
    '''
    df_train_label, df_val_label, df_test_label = data

    #Note: sklearn.precision_recall_curve returns 3 outputs: precision, recall, and threshholds. 
    #We're only interested in the second output, recall.
    xgb_train_recall = precision_recall_curve(df_train_label[cat_label], df_train_label['xgb_score'])[1]
    xgb_test_recall = precision_recall_curve(df_test_label[cat_label], df_test_label['xgb_score'])[1]
    xgb_val_recall = precision_recall_curve(df_val_label[cat_label], df_val_label['xgb_score'])[1]

    recalls: List[float] = [xgb_train_recall, xgb_test_recall, xgb_val_recall]
    return recalls
