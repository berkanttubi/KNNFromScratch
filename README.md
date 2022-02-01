You can see 2 file named knn.py and main.py. Main file controls everything , on the other hand knn file includes the functions needed for knn algorithms. You can test the algorithm manually from the algorithm_test folder.

### knn.py

+ calculate_distance: This function calculates the distance based on given data and distance metric. Here, L1 represents the manhattan distance and L2 represents the Euclidean. Since our data has 4 dimension, we have x,y,z and t here. After calculation, the distances list is returned.
+ majority_voting: It checks the majority class within the given K value. ..._labelled variables is for counting the wihch label is majority. After detecting it, the function returns the majority class.
+ knn function: It is responsible for doing the knn. It manages the functions. It first calculates the all distance values. Then it predicts the values by using majority_voting function. If predicted_label matches with test_label values true_prediction variable is increased by 1. Finally the accuracy is found by using true_prediction variable. 
+ split_train_and_validation: This function responsible for splitting the train data. First I calculate the train_fold_size based on k_fold value, so that I can choose the number of data in each fold. In the first loop I divide the labels and data and store each fold in k_folded_data variable. Then the fold for using the validation is getting by using validation_index. validation_data stores the data for validation. After that, the train folds are re-created without validation data. After the process is done, shape arrangement is done and it returns the variables.
+ cross_validation: This function is responsible for managing the cross validation. It calculates the average accuracy. It first split the data and then applies the knn functions. The accuracy is divided to k number of fold and average returns.

### main.py

​	***WARNING:*** This PROGRAM may take up to 7-8 minutes to work.

+ get_best_parameters_CV: This function does 10 K Cross Validation with different validation indexis with both L1 and L2. Then it plots the accuracies of L1 and L2 for different K values. The function returns the best k result of L1 and L2 and their best accuracies.
+ test : This function is for testing the dataset with the proper values. It calss the knn function and returns the accuracy.
+ main: This function manages the process. It chooses the more accurate K values and distance methods. Then plots them. 
