import numpy as np
from knn import calculate_distances, majority_voting, knn, split_train_and_validation, cross_validation
import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_acc(accuracies_L1,accuracies_L2,x_axis):

    plt.plot(x_axis,accuracies_L1,color = 'red', label='L1')
    plt.plot(x_axis,accuracies_L2,color = 'blue',label='L2')
    plt.legend()
    plt.xlabel("K-Values")
    plt.ylabel("Accuracy")
    plt.show()



def get_best_parameters_CV(train_set,train_labels):
    
    accuracies_L2 = list ()
    accuracies_L1 = list ()


    for i in range(len(train_set)):
        accuracies_L2.append(cross_validation(train_set, train_labels, 10, i, 'L2'))
        accuracies_L1.append(cross_validation(train_set, train_labels, 10, i, 'L1'))

    plot_acc(accuracies_L1,accuracies_L2,[i for i in range(len(train_set))])
    

    max_L1 = max(accuracies_L1)
    k_number_L1 = accuracies_L1.index(max(accuracies_L1))
    max_L2 = max(accuracies_L2)
    k_number_L2 = accuracies_L2.index(max(accuracies_L2))

    return max_L1,k_number_L1,max_L2,k_number_L2

def test(train_set,train_labels,test_set,test_labels,k,distance):
    acc = knn(train_set, train_labels, test_set, test_labels, k, distance) 
    return acc
if __name__ == '__main__':
    train_set = np.load('train_set.npy')
    train_labels = np.load('train_labels.npy')
    test_set = np.load('test_set.npy')
    test_labels = np.load('test_labels.npy')

    max_L1,k_number_L1,max_L2,k_number_L2 = get_best_parameters_CV(train_set,train_labels)

    if max_L1 > max_L2:
        k = k_number_L1
        distance = "L1"
    elif max_L2 > max_L1:
        k = k_number_L2
        distance = "L2"
    else:
        k= k_number_L1
        distance = "L1"

    print("Accuracy is: ",test(train_set,train_labels,test_set,test_labels,k,distance))