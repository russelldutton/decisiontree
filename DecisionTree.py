####################
# Import Statements
####################
# from __future__ import print_function
from math import log, floor
import sys
# import pydot

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


def train_continuous(subset, attribute_list):
    data = subset[:]  # Copy rows from args
    attrs = attribute_list[:]
    default = get_default(data)

    if len(data) == 0:
        node = {"label": default, "is_leaf": True}
        node["class"] = classes[0]
        return node
    elif is_homogenous(data, classes[0]):
        node["label"] = data[0][-1]
        node["is_leaf"] = True
        node["class"] = classes[0]
        return node
    else:
        (best_attr, best_sets) = get_best_attribute(attrs, classes[0], data)


def train_discrete(subset, attribute_list):
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
        node = {"label": default, "is_leaf": True}
        node["class"] = classes[0]
        return node
    elif is_homogenous(data, classes[0]):  # Homogenous dataset
        node = {"label": data[0][-1], "is_leaf": True}
        node["class"] = classes[0]
        return node
    else:  # Heterogenous dataset
        (best_attr, best_sets) = get_best_attribute(attrs, classes[0], data)
        node = {"label": best_attr, "is_leaf": False, "children": {}}
        # return node
        attrs.remove(best_attr)
        for o in best_sets:
            child = train_discrete(best_sets[o], attrs)
            node["children"][o] = child
        return node


def print_discrete(tree, class_name, rule_string=""):
    """
    Print tree produced by the train_discrete method
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
            print_discrete(tree["children"][key], class_name, out)


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
    best_gain = 0
    best_sets = None
    best_attr = None
    for x in attrs:
        if has_continuous and dataSpec[x] == "continuous":
            (_sets, _gain) = get_best_continuous(data, x)
        else:
            (_sets, _gain) = get_best_discrete(data, x)
        if _gain > best_gain:
            best_attr = x
            best_sets = _sets
            best_gain = _gain
    return best_attr, best_sets


def get_best_discrete(data, attribute):
    data_entropy = entropy(data, classes[0])
    subsets = partition(data, attribute)
    split_entropy = 0
    split_info = 0
    for o in subsets:
        outcome_probability = float(len(subsets[o]))/float(len(data))
        split_entropy += entropy(subsets[o], classes[0])
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
    return subsets, gain


def get_best_continuous(data, attribute):
    pass


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


def classify(tree, dataset):
    correct = 0
    for row in dataset:
        path = tree
        while path['is_leaf'] is not True:
            attr_index = attributes.index(path['label'])
            decision = row[attr_index].lower()
            path = path['children'][decision]
        if row[-1] == path['label']:
            correct += 1
    num_rows = len(dataset)
    error = floor((correct/num_rows)*100) if num_rows > 0 else -1
    return error


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
            dataSpec[attr] = vals
        elif line[-4:] == "Real":
            # Case Continuous
            attributes.append(attr)
            dataSpec[attr] = "continuous"
            pass
        else:
            # Case Class
            index = line.find(":")
            attr = line[:index]
            classes.append(attr)
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

    command = sys.argv[1]
    command = command[1:]
    if command[0] == 'c':
        has_continuous = True
    elif command[0] == 'm':
        has_missing = True
    elif command[0] == 'p':
        must_prune = True

    # graph = pydot.Dot(graph_type="graph")
    if has_continuous is True or has_missing is True or must_prune is True:
        print(("Command {s} not yet implemented").format(s=command))
    else:
        tree = train_discrete(training_dataset, attributes)
        print_discrete(tree, classes[0])
        error = classify(tree, training_dataset)
        print(error)
    # graph.write_png("graph.png")
