####################
# Import Statements
####################
# from __future__ import print_function
from math import log
import sys

########################
# Variable declarations
########################
# Dict for holding all dataSpec as keys and a list of their values
dataSpec = {}
# list of attributes in dataSpec
attributes = []
# list of the classes in dataSpec
classes = []
# list of lists containing all the data to be processed for the tree induction
training_dataset = []
test_dataset = []

########################
# Function Declarations
########################


def train_discrete(subset, attribute_list):
    """
    Discrete data algorithm
    Returns a tree in the form of:
    tree = {
        is_leaf: T/F8
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
        attrs.remove(best_attr)
        for o in best_sets:
            child = train_discrete(best_sets[o], attrs)
            node["children"][o] = child
        return node


def print_discrete(tree, class_name, start):
    """
    Print tree produced by the train_discrete method
    """
    if tree["is_leaf"] is True:
        out = "THEN {s1} is {s2}"
        print(out.format(s1=tree["class"], s2=tree["label"]))
    else:
        keys = tree["children"].keys()
        for key in keys:
            if start is True:
                print("IF", end='')
            else:
                print("AND", end='')
            out = " ( {s1} IS {s2} ) "
            print(out.format(s1=tree["label"], s2=key), end='')
            print_discrete(tree["children"][key], class_name, False)


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
    for x in attrs:
            subsets = partition(data, x)
            split_entropy = 0
            split_info = 0
            for o in subsets:
                outcome_probability = float(len(subsets[o]))/float(len(data))
                split_entropy += entropy(subsets[o], classes[0])
                split_entropy *= outcome_probability
                split_info += outcome_probability * log(outcome_probability, 2)
            gain = data_entropy - split_entropy
            gain /= (-1) * split_info
            if gain > best_gain:
                best_gain = gain
                best_sets = subsets
                best_attr = x
    best = {"sets": best_sets, "attr": best_attr}
    return best


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


def partition(set, attr):
    """
    Dataset Partition on given attribute.
    Returns dict with each subset according to attr values in dataSpec
    """
    partitionedSet = {}
    for tag in dataSpec[attr]:  # Split according to each value of attribute
        subset = []  # Temp var for holding subset
        index = attributes.index(attr)
        for row in set:
            if row[index].upper() == tag.upper():
                subset.append(row)  # Add row to subset
        partitionedSet[tag] = subset  # Store subset in dict
    return partitionedSet


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
    totalRows = len(subset)
    for value in dataSpec[classifier]:
        count = 0
        for row in subset:
            if row[index] == value:
                count += 1
        prob = float(count) / totalRows
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
        if line.endswith("}"):
            # Case Attribute
            index = line.find(":")
            attr = line[0:index]
            attributes.append(attr)
            index += 1
            vals = line[index:].lstrip(" { ").rstrip(" } ").split(", ")
            dataSpec[attr] = vals
        elif line[-4:] == "Real":
            # Case Continuous
            pass
        else:
            # Case Class
            index = line.find(":")
            attr = line[:index]
            classes.append(attr)
            index += 1
            vals = line[index:].strip().split(" ")
            dataSpec[attr] = vals

    file.close()


def read_data(filePath):
    """
    Reads in the data from data.dat to be processed in the tree induction
    """
    file = open(filePath)
    for line in file:
        training_dataset.append(line.strip().split(' '))
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
    read_spec(specPath)
    read_data(dataPath)

    # print(dataset)
    # print(get_default(dataset))
    tree = train_discrete(training_dataset, attributes)
    print(tree)
    print_discrete(tree, classes[0], True)
