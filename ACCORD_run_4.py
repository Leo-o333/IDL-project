import os
import sys
import pandas as pd
from auton_survival.preprocessing import Preprocessor
from sklearn.model_selection import ParameterGrid
from lifelines import KaplanMeierFitter
import numpy as np
#sys.path.append('./')
#sys.path.append('/Users/yushiqiu/Documents/surv-rcts')
#dataset = os.system('python3 /Users/yushiqiu/Documents/surv-rcts/dev/dev_dataset.py')
import dev.dev_dataset as dev
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc
dataset=dev.dataset


a = dataset._interventions
b = a['Main Study']

index_control = b.loc[b == dataset.control_arm].index
index_treatment = b.loc[b == dataset.treatment_arm].index
control_covariate = dataset.covariates.loc[index_control]
treatment_covariate = dataset.covariates.loc[index_treatment]

features_control = Preprocessor().fit_transform(control_covariate, cat_feats=dataset.cat_features, num_feats=dataset.num_features)
features_treatment = Preprocessor().fit_transform(treatment_covariate, cat_feats=dataset.cat_features, num_feats=dataset.num_features)
outcomes_control = dataset.outcomes.loc[index_control]
outcomes_treatment = dataset.outcomes.loc[index_treatment]


# horizons = [0.25, 0.5, 0.75]
# times = np.quantile(dataset.outcomes.time[dataset.outcomes.event==1], horizons).tolist()
# times_treatment = np.quantile(outcomes_treatment.time[outcomes_treatment.event==1], horizons).tolist()
times = [365.0,730.0,1825.0]

#stratify data
x_control, t_control, e_control = features_control.values, outcomes_control.time.values, outcomes_control.event.values

n = len(x_control)

tr_size = int(n*0.50)
vl_size = int(n*0.20)
te_size = int(n*0.30)

x_train_control, x_test_control, x_val_control = x_control[:tr_size], x_control[-te_size:], x_control[tr_size:tr_size+vl_size]
t_train_control, t_test_control, t_val_control = t_control[:tr_size], t_control[-te_size:], t_control[tr_size:tr_size+vl_size]
e_train_control, e_test_control, e_val_control = e_control[:tr_size], e_control[-te_size:], e_control[tr_size:tr_size+vl_size]

x_treatment, t_treatment, e_treatment = features_treatment.values, outcomes_treatment.time.values, outcomes_treatment.event.values

n = len(x_treatment)

tr_size = int(n*0.50)
vl_size = int(n*0.20)
te_size = int(n*0.30)

x_train_treatment, x_test_treatment, x_val_treatment = x_treatment[:tr_size], x_treatment[-te_size:], x_treatment[tr_size:tr_size+vl_size]
t_train_treatment, t_test_treatment, t_val_treatment = t_treatment[:tr_size], t_treatment[-te_size:], t_treatment[tr_size:tr_size+vl_size]
e_train_treatment, e_test_treatment, e_val_treatment = e_treatment[:tr_size], e_treatment[-te_size:], e_treatment[tr_size:tr_size+vl_size]

#Get train,val,test dataset that we need
x_train = np.concatenate((x_train_control, x_train_treatment), axis=0)
x_test = np.concatenate((x_test_control, x_test_treatment), axis=0)
x_val = np.concatenate((x_val_control, x_val_treatment), axis=0)

t_train = np.concatenate((t_train_control, t_train_treatment), axis=0)
t_test = np.concatenate((t_test_control, t_test_treatment), axis=0)
t_val = np.concatenate((t_val_control, t_val_treatment), axis=0)

e_train = np.concatenate((e_train_control, e_train_treatment), axis=0)
e_test = np.concatenate((e_test_control, e_test_treatment), axis=0)
e_val = np.concatenate((e_val_control, e_val_treatment), axis=0)

## outcomes_test will comes with the same mixing protion for both treat and ctrl
## concatenate t and e for cph and rsf
outcomes_train = np.concatenate((e_train.reshape(len(e_train), 1), t_train.reshape(len(t_train),1)), axis=1)
outcomes_train = pd.DataFrame(outcomes_train, columns=['event', 'time'])

outcomes_test = np.concatenate((e_test.reshape(len(e_test), 1), t_test.reshape(len(t_test),1)), axis=1)
outcomes_test = pd.DataFrame(outcomes_test, columns=['event', 'time'])

outcomes_val = np.concatenate((e_val.reshape(len(t_val), 1), t_val.reshape(len(e_val),1)), axis=1)
outcomes_val = pd.DataFrame(outcomes_val, columns=['event', 'time'])


param_grid = {
            # 'k' : [3, 4, 5,6],
             'k' : [3],
              'distribution' : ['LogNormal','Weibull'],
              'learning_rate' : [1e-4,1e-3],
              'layers' : [ [], [50],[50,50]],
              'elbo':[True]
             }
params = ParameterGrid(param_grid)

from auton_survival.models.dsm import DeepSurvivalMachines
from auton_survival.metrics import survival_regression_metric


def select_models(index_list,models):
    selected_models = []
    for i in range(3):
        selected_models.append(models[index_list[i]])
    return selected_models

def eval_brier(predictions,outcomes_train,outcomes_test,horizons):
    from auton_survival.metrics import survival_regression_metric
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from sksurv import metrics
    from sksurv.util import Surv
    from tabulate import tabulate
  
  
    test_uncensored_brs = []
   
    test_censored_brs = []

    from lifelines import KaplanMeierFitter

    censoring_outcomes = outcomes_train.copy()

    censoring_outcomes.event = 1 - censoring_outcomes.event

    censoring_distribution = KaplanMeierFitter().fit(censoring_outcomes.time,
                                                     censoring_outcomes.event)
   

    for i, horizon in enumerate(horizons):

        y_pos = outcomes_test.time>=horizon
        y_neg = (outcomes_test.time<horizon)&(outcomes_test.event)
    
        outcomes_uncensored  = outcomes_test.loc[y_pos|y_neg]

        y = np.zeros(len(outcomes_uncensored))

        y[outcomes_uncensored.time>=horizon] = 1
        y[outcomes_uncensored.time<horizon] = 0

    
        test_uncensored_brs.append(brier_score_loss(y, predictions[:, i][y_pos|y_neg]))
        #test_censored_eces.append(expected_calibration_error(predictions[:, 1], outcomes_test, horizon))
        weights = np.ones(outcomes_uncensored.shape[0])
        weights = censoring_distribution.predict(outcomes_uncensored.time.values)
        weights[outcomes_uncensored.time.values >= horizon] = censoring_distribution.predict(horizon)

        weights = 1. / np.clip(weights, 1e-3, 1 - 1e-3)

        test_censored_brs.append(brier_score_loss(y, predictions[:, i][y_pos | y_neg], sample_weight=weights))

    # print(test_uncensored_brs, 'brier score at evaluating horizon')

    return test_censored_brs

models_DSM = []
score_DSM = []

for param in params:
    model = DeepSurvivalMachines(k = param['k'],
                                 distribution = param['distribution'],
                                 layers = param['layers'])
    # The fit method is called to train the model
    model.fit(x_train, t_train, e_train, val_data= (x_val,t_val,e_val),iters = 200, learning_rate = param['learning_rate'],elbo=param['elbo'])
#     ### using the val data to predict the survival function
    temp_pred_DSM = model.predict_survival(x_val, times)

    score_DSM.append(eval_brier(temp_pred_DSM, outcomes_train,outcomes_val, times))
    models_DSM.append(model)
# #     ### suppose there neet to return a list of two elements, the first one is the br score, the second one is the coresponing model
# # print(score_DSM,len(models_DSM))
dsm_to_csv = pd.DataFrame(score_DSM)
file_path = "C:/Users/tiant/Documents/"
dsm_to_csv.to_csv(file_path + '/dsm_to_csv.csv')
# ####After getting the model,
socre_DSM = np.array(score_DSM)

best_model_index = np.argmin(score_DSM,axis=0)
print(best_model_index, 'index')
best_models_DSM = select_models(best_model_index,models_DSM)


## PREDICTION 
## predict the survival function using different model
def predict_survival_multi(models, x_test, times):
    for i in range(len(models)):
        
        temp_score = models[i].predict_survival(x_test, times[i])
        if i ==0:
            pred = temp_score
        
        else:
            pred = np.concatenate((pred,temp_score),axis=1)
    pred = np.array(pred)
    return pred
def predict_risk_multi(models, x_test, times):
    for i in range(len(models)):
        temp_score = models[i].predict_risk(x_test, times[i])
        if i ==0:
            pred = temp_score
        
        else:
            pred = np.concatenate((pred,temp_score),axis=1)
    pred = np.array(pred)  
            
        
    return pred

out_risk = predict_risk_multi(best_models_DSM, x_test, times)
out_survival = predict_survival_multi(best_models_DSM, x_test, times)

### EVALUATION
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc

def evaluate(predictions,out_risk, outcomes_train, outcomes_test,horizons):

    from auton_survival.metrics import survival_regression_metric
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from sksurv import metrics
    from sksurv.util import Surv
    from tabulate import tabulate
  
    test_uncensored_aucs = []
    test_uncensored_brs = []
    test_uncensored_cis = []
    #test_censored_eces = []
    test_censored_brs = []
    test_censored_auc = []
    test_censored_cis = []

    from lifelines import KaplanMeierFitter

    censoring_outcomes = outcomes_train.copy()
   
    censoring_outcomes.event = 1-censoring_outcomes.event
  
    censoring_distribution = KaplanMeierFitter().fit(censoring_outcomes.time,
                                                     censoring_outcomes.event)
   

    for i, horizon in enumerate(horizons):

        y_pos = outcomes_test.time>=horizon
        y_neg = (outcomes_test.time<horizon)&(outcomes_test.event)
    
        outcomes_uncensored  = outcomes_test.loc[y_pos|y_neg]
        y = np.zeros(len(outcomes_uncensored))

        y[outcomes_uncensored.time>=horizon] = 1
        y[outcomes_uncensored.time<horizon] = 0

    
        test_uncensored_aucs.append(roc_auc_score(y, predictions[:, i][y_pos|y_neg]))
        test_uncensored_brs.append(brier_score_loss(y, predictions[:, i][y_pos|y_neg]))
        #test_censored_eces.append(expected_calibration_error(predictions[:, 1], outcomes_test, horizon))
    
        et_test_control_uncensored = np.array([(outcomes_uncensored.iloc[i, 1], outcomes_uncensored.iloc[i, 0]) for i in range(len(outcomes_uncensored))],
                 dtype = [('e', bool), ('t', float)])
    
        et_train_control = np.array([(outcomes_train.iloc[i,1], outcomes_train.iloc[i,0]) for i in range(len(outcomes_train))],
                 dtype = [('e', bool), ('t', float)])
        
        et_test_control_censored = np.array([(outcomes_test.iloc[i, 1], outcomes_test.iloc[i, 0]) for i in range(len(outcomes_test))],
                 dtype = [('e', bool), ('t', float)])


        outcomes_uncensored  = outcomes_test.loc[y_pos|y_neg]
        out_risk_uncensored = out_risk[y_pos|y_neg]

        test_uncensored_cis.append(concordance_index_ipcw(et_train_control, et_test_control_uncensored, out_risk_uncensored[:,i], horizons[i])[0])
        
        weights = np.ones(outcomes_uncensored.shape[0])
        weights = censoring_distribution.predict(outcomes_uncensored.time.values)
        weights[outcomes_uncensored.time.values>=horizon] = censoring_distribution.predict(horizon)

        weights = 1./np.clip(weights, 1e-3, 1-1e-3)

        test_censored_auc.append(roc_auc_score(y, predictions[:, i][y_pos|y_neg], sample_weight=weights))                
        test_censored_brs.append(brier_score_loss(y, predictions[:, i][y_pos|y_neg], sample_weight=weights))
        test_censored_cis.append(concordance_index_ipcw(et_train_control, et_test_control_censored,out_risk[:,i], horizons[i])[0])
    from auton_survival.metrics import survival_regression_metric

    #test_censored_brs = survival_regression_metric('brs', outcomes['test'], predictions['TEST'], horizons, outcomes['train'])
    #test_censored_auc = survival_regression_metric('auc', outcomes['test'], predictions['TEST'], horizons, outcomes['train'])

    to_print = []
    #to_print.append(['TEST FOLD PERFORMANCE'])
    to_print.append(["BR on Uncensored Data:"] + test_uncensored_brs)
    to_print.append(["AUC on Uncensored Data:"] + test_uncensored_aucs)
    to_print.append(["C-index on Uncensored Data:"] + test_uncensored_cis)

    to_print.append(["Censoring Adjusted Brier Score:"] + list(test_censored_brs))
    to_print.append(["Censoring Adjusted AUC:"] + list(test_censored_auc))
    #to_print.append(["Censoring Adjusted ECE:"] + list(test_censored_eces))
    to_print.append(["Censoring Adjusted C-index:"] + list(test_censored_cis))

    # to_print.append(['TRAIN FOLD PERFORMANCE'])
    # train_censored_brs = metrics.brier_score(survival_train, survival_train, logistic(predictions['TRAIN']), horizons)[-1]
    # train_censored_auc = metrics.cumulative_dynamic_auc(survival_train, survival_train, 1-logistic(predictions['TRAIN']), horizons)[0]
    # to_print.append(["Censoring Adjusted Brier Score:"] + list(train_censored_brs))
    # to_print.append(["Censoring Adjusted AUC:"] + list(train_censored_auc))

    print(tabulate(to_print, headers=["METRIC"]+horizons))

def eval_train_test(best_model,x_train,x_val,outcomes_train,outcomes_val,times):
    #calculate train evaluation
    out_survival_train = predict_survival_multi(best_model, x_train, times)
    out_risk_train = 1 - out_survival_train
    print('Train Evaluation')
    evaluate(out_survival_train, out_risk_train, outcomes_train, outcomes_train, times)

    #calculate validation evaluation
    out_survival_val = predict_survival_multi(best_model, x_val, times)
    out_risk_val = 1 - out_survival_val
    print('Validation Evaluation')
    evaluate(out_survival_val, out_risk_val, outcomes_train, outcomes_val, times)




#when evaluating, pass the outcomes into evaluate()
#pass both out_survival and out_risk
## result

print('DSM')
eval_train_test(best_models_DSM,x_train,x_val,outcomes_train,outcomes_val,times)
print('Test Evaluation')
evaluate(out_survival,out_risk,outcomes_train, outcomes_test, times)

eval_train_test(best_models_DSM,x_train,x_val,outcomes_train,outcomes_val,times)

################# DCM ###############################
from auton_survival.models.dcm2 import DeepCoxMixtures
from auton_survival.models.dcm2.dcm_utilities import *
param_grid = {'k':[3,4], 'layers':[[],[50],[50,50]], 'gamma':[10],\
              'smoothing_factor':[1e-4,1e-5], 'use_activation':[True,False],\
               'random_seed':[0], 'optimizer':['SGD','Adam','RMSProp'],
             }
params = ParameterGrid(param_grid)
models_DCM =[]
score_DCM =[]
for param in params:
    model = DeepCoxMixtures(k = param['k'],layers=param['layers'],gamma = param['gamma'],smoothing_factor= param['smoothing_factor'],\
                            use_activation=param['use_activation'],random_seed=param['random_seed'])
    model.fit(x_train, t_train, e_train, val_data=(x_val,t_val, e_val),\
         iters = 100, learning_rate = 1e-4, batch_size=120,optimizer = param['optimizer'])
    temp_pred_DCM = model.predict_survival(x_val, times)
    score_DCM.append(eval_brier(temp_pred_DCM,outcomes_train,outcomes_val, times))
    models_DCM.append(model)
dcm_to_csv = pd.DataFrame(score_DCM)
file_path = "C:/Users/tiant/Documents/"
dcm_to_csv.to_csv(file_path + '/dcm_to_csv.csv')
score_DCM =np.array(score_DCM)
print(len(models_DCM))
best_model_index = np.argmin(score_DCM,axis=0)
print(best_model_index, 'index')
best_models_DCM = select_models(best_model_index,models_DCM)


out_survival_dcm = predict_survival_multi(best_models_DCM,x_test, times)
out_risk_dcm = 1-out_survival_dcm
#print(out_risk_dcm,out_survival)

print('DCM')
eval_train_test(best_models_DCM,x_train,x_val,outcomes_train,outcomes_val,times)
print('Test Evaluation')
evaluate(out_survival_dcm,out_risk_dcm,outcomes_train, outcomes_test,times)



# ################CPH#####################

# x_control, y_control = features_control, outcomes_control
#
# n = len(x_control)
#
# tr_size = int(n*0.50)
# vl_size = int(n*0.20)
# te_size = int(n*0.30)
#
# x_train_control, x_test_control, x_val_control = x_control.iloc[:tr_size], x_control.iloc[-te_size:], x_control.iloc[tr_size:tr_size+vl_size]
# y_train_control, y_test_control, y_val_control = y_control.iloc[:tr_size], y_control.iloc[-te_size:], y_control.iloc[tr_size:tr_size+vl_size]
#
# x_treatment, y_treatment = features_treatment, outcomes_treatment
#
# n = len(x_treatment)
#
# tr_size = int(n*0.50)
# vl_size = int(n*0.20)
# te_size = int(n*0.30)
#
# x_train_treatment, x_test_treatment, x_val_treatment = x_treatment.iloc[:tr_size], x_treatment.iloc[-te_size:], x_treatment.iloc[tr_size:tr_size+vl_size]
# y_train_treatment, y_test_treatment, y_val_treatment = y_treatment.iloc[:tr_size], y_treatment.iloc[-te_size:], y_treatment.iloc[tr_size:tr_size+vl_size]
#
# x_train = pd.concat((x_train_control, x_train_treatment), axis=0)
# x_test = pd.concat((x_test_control, x_test_treatment), axis=0)
# x_val = pd.concat((x_val_control, x_val_treatment), axis=0)
#
# y_train = pd.concat((y_train_control, y_train_treatment), axis=0)
# y_test = pd.concat((y_test_control, y_test_treatment), axis=0)
# y_val = pd.concat((y_val_control, y_val_treatment), axis=0)

param_grid_CPH = {'l2' : [1e-3, 1e-4]}
params_CPH = ParameterGrid(param_grid_CPH)

from auton_survival.estimators import SurvivalModel
from auton_survival.metrics import survival_regression_metric

models_cph = []
score_cph = []
for param in params_CPH:
    model = SurvivalModel('cph', random_seed=2, l2=param['l2'])

    # The fit method is called to train the model
    model.fit(x_train,outcomes_train,val_data=(x_val,outcomes_val))

    # Obtain survival probabilities for validation set and compute the Integrated Brier Score
    predictions_val = model.predict_survival(x_val, times)

    score_cph.append(eval_brier(predictions_val, outcomes_train,outcomes_val, times))
    models_cph.append(model)
# print(score_cph,len(models_cph))
# print(score_cph)
cph_to_csv = pd.DataFrame(score_cph)
file_path = "C:/Users/tiant/Documents/"
# cph_to_csv.to_csv(file_path + '/cph_to_csv.csv')
score_cph = np.array(score_cph)
# Select the best model based on the mean metric value computed for the validation set
score_cph = np.argmin(score_cph,axis=0)
# print(score_cph,'score')
model_best_cph = select_models(score_cph,models_cph)
print(score_cph,'index')
out_risk_cph = predict_risk_multi(model_best_cph,x_test, times)
out_survival_cph = 1-out_risk_cph
print('CPH')
eval_train_test(model_best_cph,x_train,x_val,outcomes_train,outcomes_val,times)
print('Test')
evaluate(out_survival_cph,out_risk_cph,outcomes_train, outcomes_test,times)


# ########################RSF!!!!!!!!!!!!!!!!!!!
param_grid_RSF = {'n_estimators' : [100, 300],
              'max_depth' : [5, 6],
              'max_features' : ['sqrt', 'log2']}
params_RSF = ParameterGrid(param_grid_RSF)
models_RSF = []
score_RSF = []
for param in params_RSF:
    model = SurvivalModel('rsf', random_seed=2, n_estimators=param['n_estimators'], max_depth=param['max_depth'], max_features=param['max_features'])
    model.fit(x_train, outcomes_train,val_data=(x_val,outcomes_val))
    predictions_val = model.predict_survival(x_val, times)
    
    score_RSF.append(eval_brier(predictions_val,outcomes_train,outcomes_val, times))
    # print(predictions_val,'predictmval')
    models_RSF.append(model)
rsf_to_csv = pd.DataFrame(score_RSF)
file_path = "C:/Users/tiant/Documents/"
# rsf_to_csv.to_csv(file_path + '/rsf_to_csv.csv')
score_RSF = np.array(score_RSF)
print(score_RSF,len(models_RSF))
best_rsf_index = np.argmin(score_RSF,axis=0)
print(best_rsf_index,'index')
best_model_rsf = select_models(best_rsf_index,models_RSF)
out_risk_RSF = predict_risk_multi(best_model_rsf,x_test, times)
out_survival_RSF = 1-out_risk_RSF
print('RSF')
eval_train_test(best_model_rsf,x_train,x_val,outcomes_train,outcomes_val,times)
print('Test')
evaluate(out_survival_RSF,out_risk_RSF,outcomes_train, outcomes_test,times)



