import csv
import sys
from math import inf,log2,sqrt

class Node:
    def __init__(self, x, subtrees):
        '''x - used feature'''
        self.x = x
        self.subtrees = subtrees

class Leaf:
    def __init__(self, label):
        self.label = label


class DecisionTreeClassifier:
    def __init__(self, max_depth = inf):
        self.max_depth = max_depth

    def fit(self, train_file):
        X, D = get_data(train_file) #X-features, D-data
        self.root = id3(D, D, X, float(self.max_depth)) #starting_node #0 is current_depth
        print("[BRANCHES]:")
        print_branches(self.root)

    def predict(self, test_file):
        X,D = get_data(test_file)
        num_of_rows = len(D)
        num_of_correct_pred = 0
        
        print("[PREDICTIONS]:", end=" ")
        predictions = []
        conf_mat_dict = {}
        used_X = set()

        for row in D:
            prediction = str(predict_row(X, row[:-1], self.root))
            real_label = str(row[-1])
            predictions.append(prediction)

            used_X.add(prediction)
            used_X.add(real_label)

            try:
                conf_mat_dict[real_label+" "+prediction] += 1
            except:
                conf_mat_dict[real_label+" "+prediction] = 1

            if prediction+" "+real_label not in conf_mat_dict.keys():
                conf_mat_dict[prediction+" "+real_label] = 0
                

            print(prediction, end=" ")
            if prediction == real_label:
                num_of_correct_pred +=1

        for i in used_X:
            for j in used_X:
                if i+" "+j not in conf_mat_dict.keys():
                    conf_mat_dict[i+" "+j] = 0

        accuracy = round(num_of_correct_pred / num_of_rows, 5)     
        print()               
        print("[ACCURACY]: {:.5f}".format(accuracy))

        print("[CONFUSION_MATRIX]:")
        num_of_columns = sqrt( len(conf_mat_dict.keys()) )

        sorted_keys = sorted( conf_mat_dict.keys() )

        for i in range(len(sorted_keys)):
            print(conf_mat_dict[sorted_keys[i]], end=" ")
            if i%num_of_columns-1 == 0:
                print()

        return predictions



def predict_row(X, row, node):

    if isinstance(node, Leaf):
        return node.label
    
    else:
        for i in range (len(X)):
            if node.x == X[i]:
                feat_value = row[i]

                new_row = row[:]
                new_X = X[:]
                del new_row[i]
                del new_X[i]

                for list1 in node.subtrees:
                    if feat_value == list1[0]:
                        label = predict_row(new_X, new_row, list1[1])
                        return label


def get_data(file_name):

    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file)

        for i, row in enumerate(csv_reader):
            if i==0: #create matrix
                X = row[:-1] # X - features
                D = []

            else:
                # D - data
                D.append(row)

    return  X, D

def entropy(data):
    labels_count_dict = {}
    num_of_rows = 0

    for row in data:
        num_of_rows += 1
        try:
            labels_count_dict[row[-1]] += 1
        except:
            labels_count_dict[row[-1]] = 1  

    ent = 0

    if len(labels_count_dict.keys()) == 1:
        return 0

    for k in labels_count_dict.keys():
        p = labels_count_dict[k]/num_of_rows
        ent -= p * log2(p)

    return ent


def calculate_IG(D_current, x, x_index): 

    current_entropy = entropy(D_current)
    sum_of_new_entropies = 0

    feature_values_dict = {}

    D = [row[:] for row in D_current]
    

    
    for i in range (len(D)):
        feature_value = D[i][x_index]
        del D[i][x_index] 

        try:
            feature_values_dict[feature_value].append(D[i])
        except:
            feature_values_dict[feature_value] = [ D[i] ]

    for k in feature_values_dict.keys():
        sum_of_new_entropies += entropy(feature_values_dict[k]) * len(feature_values_dict[k]) / len(D)

    return current_entropy - sum_of_new_entropies




def id3(D_parent, D_current, X, tree_depth):
    '''D_start - starting data
    D_current - current step data
    X - features e.g. [x1, x2, x3]
    y - labels'''

    labels_count_dict = {}

    if tree_depth==0:
        for row in D_current: 
            try:
                labels_count_dict[row[-1]] += 1
            except:
                labels_count_dict[row[-1]] = 1

        max_num_label = -1
        for label in labels_count_dict.keys():
            if labels_count_dict[label] > max_num_label or labels_count_dict[label] == max_num_label and label<most_used_label:
                most_used_label = label # v
                max_num_label = labels_count_dict[label]

        return Leaf(most_used_label)
    
    if len(D_current)==0:
        
        for row in D_parent: 
            try:
                labels_count_dict[row[-1]] += 1
            except:
                labels_count_dict[row[-1]] = 1

        max_num_label = -1
        for label in labels_count_dict.keys():
            if labels_count_dict[label] > max_num_label or labels_count_dict[label] == max_num_label and label<most_used_label:
                most_used_label = label # v
                max_num_label = labels_count_dict[label]

        return Leaf(most_used_label)
    


    for row in D_current: 
        try:
            labels_count_dict[row[-1]] += 1
        except:
            labels_count_dict[row[-1]] = 1

    max_num_label = -1
    for label in labels_count_dict.keys():
        if labels_count_dict[label] > max_num_label or labels_count_dict[label] == max_num_label and label<most_used_label:
            most_used_label = label # v
            max_num_label = labels_count_dict[label]

    different_labels_exist = False
    for list1 in D_current:
        if list1[-1] != most_used_label:
            different_labels_exist = True
            break


    if len(X)==0 or different_labels_exist == False:
        return Leaf(most_used_label)
    
    #calculate x with highest IG
    highest_IG = -inf
    for i in range(len(X)): 
        new_IG = calculate_IG(D_current, X[i], i)
        print("IG({}): {}".format( X[i],round(new_IG,4) ), end=" ")
        
        if new_IG > highest_IG or new_IG == highest_IG and X[i] < best_x:
            highest_IG = new_IG
            best_x = X[i]
            best_x_index = i
    
    print("")

    
    subtrees = []
    feature_values_dict = {}
    D = [row[:] for row in D_current]

    #
    for i in range (len(D)):
        feature_value = D[i][best_x_index]
        del D[i][best_x_index] 

        try:
            feature_values_dict[feature_value].append(D[i])
        except:
            feature_values_dict[feature_value] = [ D[i] ]
    
    X_new = X[:]
    del X_new[best_x_index]

    for k in feature_values_dict.keys():
        t = id3(D_parent, feature_values_dict[k], X_new, tree_depth-1)

        subtrees.append([k, t])

    return Node(best_x, subtrees)




def print_branches(node, i=1, print_list=[]):
    if isinstance(node, Leaf):
        
        for i in range(len(print_list)):
            print(print_list[i], end=" ")
        print(node.label)

    else:
        for list1 in node.subtrees: #list1 = [v,t]
            new_print_list = print_list.copy()

            new_print_list.append("{}:{}={}".format(i, node.x, list1[0]))
            print_branches(list1[1], i+1, new_print_list)
 
    return

#main

train_file = sys.argv[1]
test_file = sys.argv[2]

if len(sys.argv) > 3:
    tree_depth = sys.argv[3]
    model = DecisionTreeClassifier(tree_depth)

else:
    model = DecisionTreeClassifier()

model.fit(train_file)

predictions = model.predict(test_file)