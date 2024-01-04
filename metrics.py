from collections import defaultdict
import numpy as np
from mlxtend.plotting import plot_confusion_matrix
from matplotlib import pyplot as plt

POSITIVE = 1
NEGATIVE = 0            

def get_metrics_from_labels(phrases_of_computed_labels : list[list[int]], 
                                    phrases_of_effective_labels : list[list[int]]) -> dict:
    results = defaultdict(lambda : [])
    for computed_labels, effective_lables\
            in zip(phrases_of_computed_labels, phrases_of_effective_labels):
        confusion_matrix = get_confusion_matrix(computed_labels, effective_lables) 
        precision = get_precision_from_confusion_matrix(confusion_matrix)
        recall = get_recall_from_confusion_matrix(confusion_matrix) 
        accuracy = get_accuracy_from_confusion_matrix(confusion_matrix)
        results["confusion_matrix"] +=[confusion_matrix]
        results["precision"] += [precision]
        results["recall"] += [recall]
        results["accuracy"] += [accuracy]
        results["f1-score"] += [(recall * precision *2 )/ (recall + precision)]
    return results 
             
def get_confusion_matrix(computed_labels : list[int], effective_labels: list[int])-> np.ndarray:
    confusion_matrix = np.zeros((2,2))
    for computed_label, effective_label in zip(computed_labels, effective_labels):
        confusion_matrix[effective_label][computed_label] += 1
    return confusion_matrix
             
 
def get_precision_from_confusion_matrix(confusion_matrix : np.ndarray): 
    true_positive  = confusion_matrix[POSITIVE][POSITIVE]
    false_positive = confusion_matrix[NEGATIVE][POSITIVE]
    return true_positive/(true_positive + false_positive)

def get_recall_from_confusion_matrix(confusion_matrix : np.ndarray):
    true_positive = confusion_matrix[POSITIVE][POSITIVE]
    false_negative = confusion_matrix[POSITIVE][NEGATIVE]
    return true_positive/(true_positive + false_negative)
 
def get_accuracy_from_confusion_matrix(confusion_matrix):
    true_positive = confusion_matrix[POSITIVE][POSITIVE]
    true_negative = confusion_matrix[NEGATIVE][NEGATIVE]
    false_positive = confusion_matrix[NEGATIVE][POSITIVE]
    false_negative = confusion_matrix[POSITIVE][NEGATIVE]
    return true_positive + true_negative / (false_positive + false_negative)

def print_confusion_matrix(confusion_matrix : np.ndarray):
    
    fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix, 
                                    figsize=(6, 6), 
                                    cmap=plt.cm.Blues)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig("/result/confusion_matrix.png")
    plt.show()