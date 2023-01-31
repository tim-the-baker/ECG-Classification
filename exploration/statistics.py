"""
File implementing many different statistics measures. Each function in the file takes in
a confusion matrix as input and outputs a statistic metric (e.g., precision, recall, etc.)
Terms are explained assuming a medical context.
"""
import csv, torch, os
import numpy as np

def ACC(matrix):
    """
    Accuracy measures how often the classification was correct. It doesn't account for prevalence of positives vs.
    negatives.

    In terms of confusion matrix, the formula is:
    ACC = (TP + TN) / (TP + FP + TN + FN)

    :param matrix: 2x2 confusion matrix
    :return: accuracy
    """
    (tp, fn), (fp, tn) = matrix

    if tp + tn + fp + tn == 0:  # edge case where matrix is all zeros for whatever reason
        acc = 1
    else:
        acc = (tp + tn) / (tp + fp + tn + fn)

    return acc


def PPV(matrix):
    """
    Positive predicted value (aka precision) measures how often positive classifications are correct. In other words,
    given that a patient tests positive, PPV is the likelihood that they do have the disease.

    In terms of the confusion matrix, PPV is:
    PPV = TP / (TP + FP)

    If there are no positive patients at all, then PPV=1. If there are some positive patients, but no positive tests,
    then PPV=0.
    :param matrix: 2x2 confusion matrix
    :return: positive predicted value
    """
    (tp, fn), (fp, tn) = matrix

    if tp + fn == 0:  # if there are no members of the positive class, then PPV=1
        ppv = 1
    elif tp + fp == 0:  # if there are positive members, but not positive classifications, then PPV=0
        ppv = 0
    else:
        ppv = tp / (tp + fp)

    return ppv


def NPV(matrix):
    """
    Negative predicted value measures how often negative classifications are correct. In other words,
    given a patient tests negative, NPV is the likelihood that they don't have the disease.

    In terms of the confusion matrix, NPV is:
    NPV = TN / (TN + FN)

    If there are no negative patients at all, then NPV=1. If there are some negative samples, but no negative
    classifications, then NPV=0.
    :param matrix: 2x2 confusion matrix
    :return: negative predicted value
    """
    (tp, fn), (fp, tn) = matrix

    if tn + fp == 0:  # if there are no members of the negative class, then NPV=1
        npv = 1
    elif tn + fn == 0:  # if there are negative members, but not negative classifications, then NPV=0
        npv = 0
    else:
        npv = tn / (tn + fn)

    return npv


def sensitivity(matrix):
    """
    Sensitivity (aka recall or true positive rate) measures how often a positive member is classified as positive.
    In other words, given a patient has a disease, sensitivity is the likelihood that they test positive.

    In terms of the confusion matrix, sensitivity is:
    SEN = TP / (TP + FN)

    :param matrix: 2x2 confusion matrix
    :return: sensitivity
    """
    (tp, fn), (fp, tn) = matrix

    if tp + fn == 0:  # if there are no members of the positive class, then sensitivity should be 1
        sen = 1
    else:
        sen = tp / (tp + fn)

    return sen


def specificity(matrix):
    """
    Specificity (aka selectivity or true negative rate) measures how often a negative member is classified as negative.
    In other words, given a patient does not have a disease, specificity is the likelihood that they test negative.

    In terms of the confusion matrix, specificity is:
    SEN = TN / (TN + FP)

    :param matrix: 2x2 confusion matrix
    :return: specificity
    """
    (tp, fn), (fp, tn) = matrix

    if tn + fp == 0:  # if there are no members of the negative class, then specificity should be 1
        spec = 1
    else:
        spec = tn / (tn + fp)

    return spec


def BAC(matrix):
    """
    Balanced accuracy combines sensitivity and specificity into a single metric.

    In terms of the confusion balanced accuracy is:
    BAC = 0.5 * (TP/(TP + FN) + TN/(TN + FP)) = 0.5 * (SEN + SPEC)
    :param matrix: 2x2 confusion matrix
    :return: balanced accuracy
    """

    return (sensitivity(matrix) + specificity(matrix))/2


# TODO: F1, FB
