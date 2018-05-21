####################
# Import Statements
####################
# from __future__ import print_function
from math import log
from math import floor
from random import randint
import sys
# import pydot

########################
# Variable declarations
########################
# Dict for holding all dataSpec as keys and a list of their values
dataSpec = {}
# list of attributes in dataSpec
attributes = []
is_continuous_attr = []
# list of the classes in dataSpec
classes = []
# list of lists containing all the data to be processed for the tree induction
training_dataset = []
# data to test tree classification
test_dataset = []
# condition to account for missing values
has_missing = False
# condition to account for continuous data
has_continuous = False
# condition whether the tree should be pruned after being created
must_prune = False

########################
# Function Declarations
########################


def train_tree(subset, attribute_list):
    """
    Discrete data algorithm
    Returns a tree in the form of:
    tree = {
        label: <string>
        is_leaf: T/F
        class: <class> # Only if is_leaf is true
        children = { # Only if is_leaf is false will children have values
                <label>: <tree>
            }
    }
    """
    data = subset[:]
    attrs = attribute_list[:]
    default = get_default(data)

    if len(data) == 0:   # If empty dataset
        node = {"label": default, "is_leaf": True, "children": {}}
        node["class"] = classes[0]
        return node
    elif is_homogenous(data, classes[0]):  # Homogenous dataset
        node = {"label": data[0][-1], "is_leaf": True, "children": {}}
        node["class"] = classes[0]
        return node
    else:  # Heterogenous dataset
        best = get_best_attribute(attrs, classes[0], data)
        best_sets = best["sets"]
        best_attr = best["attr"]
        node = {"label": best_attr, "is_leaf": False, "children": {}}
        # return node
        attr_remove_value = best_attr.split(" ")[0]
        attrs.remove(attr_remove_value)
        for o in best_sets:
            child = train_tree(best_sets[o], attrs)
            o = o if is_real(o) else o.lower()
            node["children"][o] = child
        return node


def print_tree(tree, class_name, rule_string=""):
    """
    Print tree produced by the train_tree
    method
    """
    if tree["is_leaf"] is True:
        rule_string += "THEN {s1} is {s2}"
        print(rule_string.format(s1=tree["class"], s2=tree["label"]))
        # draw(tree["label"], tree[""])
    else:
        keys = tree["children"].keys()
        for key in keys:
            out = rule_string
            if rule_string is "":
                out = "IF"
            else:
                out += "AND"
            temp_string = " ( {s1} IS {s2} ) "
            out += temp_string.format(s1=tree["label"], s2=key)
            # if tree["children"] is not {}:
            #     if tree["children"][key]["is_leaf"] is True:
            #         draw(tree["label"], tree["label"] + "_"
            #              + tree["children"][key]["label"], key)
            #     else:
            #         draw(tree["label"],
            #              tree["children"][key]["label"], key)
            print_tree(tree["children"][key], class_name, out)


# def draw(parent, vert, label):
#     edge = pydot.Edge(parent, vert, label=label)
#     graph.add_edge(edge)


def is_homogenous(data, classifier):
    """
    Test whether the data set data is homogenous
    """
    tag = data[0][-1]
    for row in data:
        if tag != row[-1]:
            return False
    return True


def get_best_attribute(attrs, class_val, data):
    """
    Get the best attribute to split the data set on
    """
    data_entropy = entropy(data, classes[0])
    best_gain = 0
    best_sets = None
    best_attr = None
    split_entropy = 0
    split_info = 0
    for x in attrs:
        if has_continuous and dataSpec[x] == "continuous":
            value_list = get_unique_list(data, x)
            best_continuous_gain = 0
            upper_bound = len(value_list) - 1
            for index in range(upper_bound):
                split_entropy = 0
                split_info = 0
                x_one = value_list[index]
                x_two = value_list[index + 1]
                threshold = (x_one + x_two) / 2
                subsets = partition(data, x, threshold)
                len_data = float(len(data))
                for s in subsets:  # subsets here have 2 lists
                    len_subset = float(len(subsets[s]))
                    outcome_probability = len_subset/len_data
                    split_entropy += entropy(subsets[s], classes[0])
                    split_entropy *= outcome_probability
                    temp = 0
                    if outcome_probability > 0:
                        temp += outcome_probability
                        temp *= log(outcome_probability, 2)
                        split_info += temp
                    else:
                        split_info += 0
                gain = data_entropy - split_entropy
                if split_info < 0:
                    gain /= (-1.0 * split_info)
                if gain > best_continuous_gain:
                    best_continuous_gain = gain
                    best_attr = threshold
        else:
            subsets = partition(data, x)
            for s in subsets:
                len_subset = float(len(subsets[s]))
                len_data = float(len(data))
                outcome_probability = len_subset/len_data
                split_entropy += entropy(subsets[s], classes[0])
                split_entropy *= outcome_probability
                temp = 0
                if outcome_probability > 0:
                    temp += outcome_probability
                    temp *= log(outcome_probability, 2)
                    split_info += temp
                else:
                    split_info += 0
            gain = data_entropy - split_entropy
            if split_info > 0:
                gain /= -1.0 * split_info
        if gain > best_gain:
            best_gain = gain
            best_sets = subsets
            best_attr = x
            if dataSpec[x] == "continuous" and has_continuous:
                best_attr += " <= " + str(threshold)
    best = {"sets": best_sets, "attr": best_attr}
    return best


def get_unique_list(data, attr):
    new_list = []
    for row in data:
        index = attributes.index(attr)
        val = int(row[index])
        if val not in new_list:
            new_list.append(val)
    new_list.sort()
    return new_list


def get_default(subset):
    """
    Get the majority class
    """
    counts = []
    classValues = dataSpec[classes[0]]
    for i in range(len(classValues)):
        counts.append(0)
        for row in subset:
            if row[-1] == classValues[i]:
                counts[i] += 1
    return classValues[counts.index(max(counts))]


def partition(data, split, threshold=None):
    """
    Dataset Partition on given attribute.
    Returns dict with each subset according to attr values in dataSpec
    """
    partitionedSet = {}
    if has_continuous and is_real(threshold):
        partitionedSet[0] = []
        partitionedSet[1] = []
        for row in data:
            index = 0 if row[attributes.index(split)] < threshold else 1
            partitionedSet[index].append(row)
    else:
        # Split according to each value of attribute
        for tag in dataSpec[split]:
            subset = []  # Temp var for holding subset
            index = attributes.index(split)
            for row in data:
                if not is_real(row[index]) and row[index].upper() == tag.upper():
                    subset.append(row)  # Add row to subset
            partitionedSet[tag] = subset  # Store subset in dict
    return partitionedSet


def classify(tree, dataset):
    correct = 0
    for row in dataset:
        path = tree
        while path['is_leaf'] is not True:
            path_label = path["label"].split()[0]
            attr_index = attributes.index(path_label)
            if has_continuous and is_real(row[attr_index]):
                threshold = float(path["label"].split()[-1])
                decision = 0 if row[attr_index] <= threshold else 1
            else:
                row[attr_index] = row[attr_index].lower()
                decision = row[attr_index]
            path = path['children'][decision]
        if row[-1] == path['label']:
            correct += 1
    num_rows = len(dataset)
    error = floor((correct/num_rows)*100)
    return error


def is_real(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def entropy(subset, classifier):
    """
    Calculate entropy of a set.
    Parameters are
    subset to calculate the entropy on, and
    the classifier to use
    Returns the entropy of the data set
    """
    # Calculate index for class field. Usually -1.
    # But can accomodate multiple classes
    index = - len(classes) - classes.index(classifier)
    entropy = 0
    total_rows = len(subset)
    for value in dataSpec[classifier]:
        count = 0
        for row in subset:
            if row[index] == value:
                count += 1
        if total_rows > 0:
            prob = float(count) / total_rows
        else:
            prob = 0
        if prob == 0:
            entropy += 0
        else:
            entropy += prob * log(prob, 2)
    return entropy * -1


def read_spec(filePath):
    """
    Reads in and interprets the data.spec file that details the data properties
    for the algorithm
    """
    file = open(filePath)
    # spec = file.read()

    for line in file:
        line = line.strip()
        index = line.find(":")
        attr = line[0:index]
        index += 1
        if line.endswith("}"):
            # Case Attribute
            attributes.append(attr)
            vals = line[index:].lstrip(" { ").rstrip(" } ").split(", ")
            for val in vals:
                val = val.lower()
            dataSpec[attr] = vals
        elif line[-4:] == "Real":
            # Case continuous
            attributes.append(attr)
            dataSpec[attr] = "continuous"
            pass
        else:
            # Case Class
            classes.append(attr)
            vals = line[index:].strip().split()
            dataSpec[attr] = vals
        index += 1

    file.close()


def read_data(filePath):
    """
    Reads in the data from data.dat to be processed in the tree induction
    """
    file = open(filePath)
    for line in file:
        training_dataset.append(line.strip().split())
        for a in attributes:
            if dataSpec[a] == 'continuous' and has_continuous:
                val = int(training_dataset[-1][attributes.index(a)])
                training_dataset[-1][attributes.index(a)] = val

    num_records = len(training_dataset)
    num_test = int(floor(0.3 * num_records))
    for i in range(0, num_test):
        test_dataset.append(training_dataset.pop(randint(0, num_records - 1)))
        num_records -= 1
    file.close()


################
# "Main Method"
################
if __name__ == '__main__':
    specPath = "data/data.spec"
    dataPath = "data/data.dat"
    if len(sys.argv) > 3:
        specPath = sys.argv[2]
        dataPath = sys.argv[3]

    command = sys.argv[1]
    command = command[1:]
    if command[0] == 'c':
        has_continuous = True
    elif command[0] == 'm':
        has_missing = True
    elif command[0] == 'p':
        must_prune = True

    read_spec(specPath)
    read_data(dataPath)

    # graph = pydot.Dot(graph_type="graph")
    if has_missing or must_prune:
        print(("Command {s} not yet implemented").format(s=command))
    else:
        # print(dataSpec)
        tree = train_tree(training_dataset, attributes)
        # print_tree(tree, classes[0])
        print(tree)
        # print(attributes)
        # print(training_dataset[0])
        error = classify(tree, training_dataset)
        print("Training Set Correctness: ", error, "%")
        # error = classify(tree, test_dataset)
        # print("Test Set Correctness:", error, "%")
    # graph.write_png("graph.png")
