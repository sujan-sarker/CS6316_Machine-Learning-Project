from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np



def final_performance(per_dict):
    final_perf = {'accuracy':[],'precision_macro':[], 'precision_micro':[], 'precision_weighted':[], 'recall_macro':[], 'recall_micro':[], 'recall_weighted':[],'f1_macro':[], 'f1_micro':[], 'f1_weighted':[], 'roc_auc':[]}
    for perf in per_dict:
        for key in perf:
            final_perf[key].append(perf[key])
    
    temp = {}
    for key in final_perf:
        temp[key] = np.mean(final_perf[key])
    return temp
        

def calculate_performance(y_pred, y_true, is_sigmoid=False):
#     if type(y_true[0].__type__()) != 'int':
#         y_true = np.argmax(y_true, axis=1)
    temp = y_pred
    
    if is_sigmoid:
        y_pred = np.round(y_pred)
    else:
        y_pred = np.argmax(y_pred, axis=1)
    report = {}
    report['accuracy'] = accuracy_score(y_pred, y_true)
    
    report['precision_macro'] = precision_score(y_true, y_pred, average='macro')
    report['precision_micro'] = precision_score(y_true, y_pred, average='micro')
    report['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
    
    report['recall_macro'] = recall_score(y_true, y_pred, average='macro')
    report['recall_micro'] = recall_score(y_true, y_pred, average='micro')
    report['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
    
    report['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    report['f1_micro'] = f1_score(y_true, y_pred, average='micro')
    report['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    
    report['roc_auc'] = roc_auc_score(y_true, temp, multi_class='ovr')
    
    return report
    
    