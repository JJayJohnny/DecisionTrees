import numpy as np

def calc_total_entropy(xTrain, yTrain, class_list):
    total_row = xTrain.shape[0]
    total_entr = 0

    for c in class_list:
        total_class_count = xTrain.loc[yTrain == c].shape[0]
        total_class_entr = - (total_class_count / total_row) * np.log2(total_class_count / total_row)
        total_entr += total_class_entr

    return total_entr


def calc_entropy(xTrain, yTrain, class_list):
    class_count = xTrain.shape[0]
    entropy = 0

    for c in class_list:
        label_class_count = xTrain.loc[yTrain == c].shape[0]  # row count of class c
        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count / class_count
            entropy_class = - probability_class * np.log2(probability_class)  # entropy
        entropy += entropy_class
    return entropy


def calc_info_gain(feature_name, xTrain, yTrain, class_list):
    feature_value_list = xTrain[feature_name].unique()  # unqiue values of the feature
    total_row = xTrain.shape[0]
    feature_info = 0.0

    for feature_value in feature_value_list:
        feature_value_data = xTrain[xTrain[feature_name] == feature_value]  # filtering rows with that feature_value
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calc_entropy(feature_value_data, yTrain, class_list)
        feature_value_probability = feature_value_count / total_row
        feature_info += feature_value_probability * feature_value_entropy  # calculating information of the feature value

    return calc_total_entropy(xTrain, yTrain, class_list) - feature_info  # calculating information gain


def find_most_informative_feature(xTrain, yTrain, class_list):
    feature_list = xTrain
    max_info_gain = -1
    max_info_feature = None

    for feature in feature_list:
        feature_info_gain = calc_info_gain(feature, xTrain, yTrain, class_list)
        if max_info_gain < feature_info_gain:  # selecting feature name with highest information gain
            max_info_gain = feature_info_gain
            max_info_feature = feature

    return max_info_feature


def generate_sub_tree(feature_name, xTrain, yTrain, class_list):
    feature_value_count_dict = xTrain[feature_name].value_counts(
        sort=False)
    tree = {}

    for feature_value, count in feature_value_count_dict.items():
        feature_value_data = xTrain[xTrain[feature_name] == feature_value]

        assigned_to_node = False  # flag for tracking feature_value is pure class or not
        for c in class_list:
            class_count = feature_value_data.loc[yTrain == c].shape[0]

            if class_count == count:
                tree[feature_value] = c
                xTrain = xTrain[xTrain[feature_name] != feature_value]
                assigned_to_node = True
        if not assigned_to_node:  # not pure class
            tree[feature_value] = "?"  # as feature_value is not a pure class, it should be expanded further,

    return tree, xTrain


def make_tree(root, prev_feature_value, xTrain, yTrain, class_list):
    if xTrain.shape[0] != 0:
        max_info_feature = find_most_informative_feature(xTrain, yTrain, class_list)
        tree, train_data = generate_sub_tree(max_info_feature, xTrain, yTrain, class_list)
        next_root = None

        if prev_feature_value != None:
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature] = tree
            next_root = root[prev_feature_value][max_info_feature]
        else:
            root[max_info_feature] = tree
            next_root = root[max_info_feature]

        for node, branch in list(next_root.items()):
            if branch == "?":
                feature_value_data = train_data[train_data[max_info_feature] == node]
                make_tree(next_root, node, feature_value_data, yTrain, class_list)
    return root



def id3(xTrainO, yTrainO):
    xTrain = xTrainO.copy()
    yTrain = yTrainO.copy()
    class_list = yTrain.unique()
    tree = make_tree({}, None, xTrain, yTrain, class_list)
    return tree

def predict(tree, instance):
    if not isinstance(tree, dict):
        return tree
    else:
        root_node = next(iter(tree))
        feature_value = instance[root_node]
        if feature_value in tree[root_node]:
            return predict(tree[root_node][feature_value], instance)
        else:
            return None


def evaluate(tree, xTest, yTest):
    correct_preditct = 0
    wrong_preditct = 0
    for index in range(xTest.shape[0]):
        result = predict(tree, xTest.iloc[index])
        if result == yTest.iloc[index]:
            correct_preditct += 1
        else:
            wrong_preditct += 1
    accuracy = correct_preditct / (correct_preditct + wrong_preditct)
    return accuracy