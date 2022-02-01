import numpy as np
import math
from random import randrange

def calculate_distances(train_data, test_instance, distance_metric):
    """
    Calculates Manhattan (L1) / Euclidean (L2) distances between test_instance and every train instance.
    :param train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data.
    :param test_instance: A (D, ) shaped numpy array.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: An (N, ) shaped numpy array that contains distances.
    """
    distances = []
    temp = 0
    if (distance_metric=="L1"):
        for x,y in train_data:
            distances.append(abs(test_instance[0]-x) + abs(test_instance[1]-y))
            
    elif (distance_metric=="L2"):
        for x,y in train_data:
            temp = (test_instance[0]-x)**2 + (test_instance[1]-y)**2
            distances.append(np.sqrt(temp))
            
    return distances
            


def majority_voting(distances, labels, k):
    """
    Applies majority voting. If there are more then one major class, returns the smallest label.
    :param distances: An (N, ) shaped numpy array that contains distances
    :param labels: An (N, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :return: An integer. The label of the majority class.
    """

    zero_labelled = 0
    one_labelled = 0
    for i in range(k):
        if (labels[np.argmin(distances)] == 1):
            one_labelled+=1
        else:
            zero_labelled+=1

        distances[np.argmin(distances)] = 1000
    
    if zero_labelled > one_labelled:
        return 0
    elif zero_labelled < one_labelled:
        return 1
    else:
        return 0



def knn(train_data, train_labels, test_data, test_labels, k, distance_metric):
    """
    Calculates accuracy of knn on test data using train_data.
    :param train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param train_labels: An (N, ) shaped numpy array that contains labels
    :param test_data: An (M, D) shaped numpy array where M is the number of examples
    and D is the dimension of the data
    :param test_labels: An (M, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: A float. The calculated accuracy.
    """
    distances = []
    predicted_label = []
    true_prediction = 0
    for i in range(len(test_data)):
        distances.append(calculate_distances(train_data, test_data[i], distance_metric))

    for i in range(len(distances)):
        predicted_label.append(majority_voting(distances[i], train_labels, k))
        
    for i in range(len(predicted_label)):
        if predicted_label[i] == test_labels [i]:
            true_prediction +=1


    return (true_prediction / len(predicted_label))   


def split_train_and_validation(whole_train_data, whole_train_labels, validation_index, k_fold):
    """
    Splits training dataset into k and returns the validation_indexth one as the
    validation set and others as the training set. You can assume k_fold divides N.
    :param whole_train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param whole_train_labels: An (N, ) shaped numpy array that contains labels
    :param validation_index: An integer. 0 <= validation_index < k_fold. Specifies which fold
    will be assigned as validation set.
    :param k_fold: The number of groups that the whole_train_data will be divided into.
    :return: train_data, train_labels, validation_data, validation_labels
    train_data.shape is (N-N/k_fold, D).
    train_labels.shape is (N-N/k_fold, ).
    validation_data.shape is (N/k_fold, D).
    validation_labels.shape is (N/k_fold, ).
    """
    train_fold_size = int(len(whole_train_data) / k_fold)
    k_folded_data = list()
    k_folded_label = list()
    train_data = list()
    train_labels = list()
    for i in range(0,len(whole_train_data),train_fold_size):  
        k_folded_data.append(whole_train_data[i:i+train_fold_size])
        k_folded_label.append(whole_train_labels[i:i+train_fold_size])

    validation_data = k_folded_data[validation_index]
    validation_label = k_folded_label[validation_index]

    for i in range(len(k_folded_data)):
        if i != validation_index:
            train_data.append(k_folded_data[i])
            train_labels.append(k_folded_label[i])
    
    train_data = np.asarray(train_data)
    train_data = train_data.reshape((train_data.shape[0]*train_data.shape[1],train_data.shape[2]))
    train_label = np.asarray(train_labels)
    train_label = train_label.flatten()
    
    return train_data,train_label,validation_data,validation_label




def cross_validation(whole_train_data, whole_train_labels, k_fold, k, distance_metric):
    """
    Applies k_fold cross-validation and averages the calculated accuracies.
    :param whole_train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param whole_train_labels: An (N, ) shaped numpy array that contains labels
    :param k_fold: An integer.
    :param k: An integer. The number of nearest neighbor to be selected.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: A float. Average accuracy calculated.
    """
    
    acc=0
    for i in range(k_fold):
        train_data,train_label,validation_data,validation_label = split_train_and_validation(whole_train_data,whole_train_labels,i,k_fold)
        acc+=knn(train_data,train_label,validation_data,validation_label,k,distance_metric)

    return acc/k_fold