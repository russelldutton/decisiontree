####################
# Import Statements
####################
from math import log, floor
from random import randint
import sys


########################
# Variable declarations
########################
# Dict for holding all data_spec as keys and a list of their values
data_spec = {}
# list of attributes in data_spec
attributes = []
# list of the classes in data_spec
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
    node = {}

    if len(data) == 0:
        node["label"] = default
        node["is_leaf"] = True
        node["class"] = classes[0]
        return node
    elif is_homogenous(data, classes[0]):
        node["label"] = data[0][-1]
        node["is_leaf"] = True
        node["class"] = classes[0]
        return node
    else:
        (best_attr,
         best_sets, threshold) = get_best_attribute(attrs, classes[0], data)
        node["label"] = best_attr
        node["is_leaf"] = False
        node["children"] = {}
        if threshold is None:
            # attrs.remove(best_attr)
            for o in best_sets:
                child = train_continuous(best_sets[o], attrs)
                node["children"][o] = child
        else:
            node["threshold"] = threshold
            # attrs.remove(best_attr)
            for o in best_sets:
                child = train_continuous(best_sets[o], attrs)
                node["children"][o] = child
        return node


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
        (best_attr, best_sets,
         _t) = get_best_attribute(attrs, classes[0], data)
        node = {"label": best_attr, "is_leaf": False, "children": {}}
        attrs.remove(best_attr)
        for o in best_sets:
            child = train_discrete(best_sets[o], attrs)
            node["children"][o] = child
        return node


def print_discrete(tree, class_name, rule_string=""):
    """
    Print tree produced by the train_discrete function
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
            print_discrete(tree["children"][key], class_name, out)


def print_continuous(tree, class_name, rule_string=""):
    """
    Print tree produced by the train_continuous function
    """
    if tree["is_leaf"]:
        rule_string += "THEN {s1} is {s2}"
        print(rule_string.format(s1=tree["class"], s2=tree["label"]))
    else:
        keys = tree["children"].keys()
        for key in keys:
            out = rule_string
            if rule_string is "":
                out = "IF"
            else:
                out += "AND"
            temp_string = " ( {s1} IS {s2} ) "
            if 'threshold' in tree:
                threshold = tree["label"] + " < " + str(tree["threshold"])
                out += temp_string.format(s1=threshold, s2=(key == 0))
            else:
                out += temp_string.format(s1=tree["label"], s2=key)
            print_continuous(tree["children"][key], class_name, out)


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
    best_threshold = None
    _threshold = None
    for x in attrs:
        if has_continuous and data_spec[x] == "continuous":
            (_sets, _gain, _threshold) = get_best_continuous(data, x)
        else:
            (_sets, _gain) = get_best_discrete(data, x)
            _threshold = None
        if _gain > best_gain:
            best_attr = x
            best_sets = _sets
            best_gain = _gain
            best_threshold = _threshold
    return best_attr, best_sets, best_threshold


def get_best_discrete(data, attribute):
    data_entropy = entropy(data, classes[0])
    subsets = partition(data, attribute)
    split_entropy = 0
    split_info = 0
    for o in subsets:
        outcome_probability = float(len(subsets[o]))/float(len(data))
        subset_entropy = entropy(subsets[o], classes[0])
        subset_entropy *= outcome_probability
        split_entropy += subset_entropy
        temp = 0
        if outcome_probability > 0:
            temp += outcome_probability
            temp *= log(outcome_probability, 2)
        split_info += temp
    gain = data_entropy - split_entropy
    if split_info != 0:
        gain /= -1.0 * split_info
    return subsets, gain


def get_best_continuous(data, attribute):
    data_entropy = entropy(data, classes[0])
    best_sets = None
    best_gain = 0
    best_threshold = None
    unique_list = get_unique_list(data, attributes.index(attribute))
    for index in range(len(unique_list) - 1):
        split_entropy = 0
        split_info = 0
        subset_entropy = 0
        x_one = unique_list[index]
        x_two = unique_list[index + 1]
        threshold = round(float((x_one + x_two) / 2), 5)
        subsets = partition_continuous(data, attribute, threshold)
        for o in subsets:
            subset_rows = float(len(subsets[o]))
            ttl_rows = float(len(data))
            outcome_probability = subset_rows/ttl_rows
            subset_entropy = entropy(subsets[o], classes[0])
            subset_entropy *= outcome_probability
            split_entropy += subset_entropy
            temp = 0
            if outcome_probability > 0:
                temp += outcome_probability
                temp *= log(outcome_probability, 2)
            split_info += temp
        gain = data_entropy - split_entropy
        # if split_info < 0:
        #     gain /= -1.0 * split_info
        if gain > best_gain:
            best_threshold = threshold
            best_gain = gain
            best_sets = subsets
    return best_sets, best_gain, best_threshold


def get_unique_list(data, row_index):
    values = []
    for row in data:
        if row[row_index] not in values:
            values.append(float(row[row_index]))
    values.sort()
    return values


def is_real(val):
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False


def get_default(subset):
    """
    Get the majority class
    """
    counts = []
    classValues = data_spec[classes[0]]
    for i in range(0, len(classValues)):
        counts.append(0)
        for row in subset:
            if row[-1] == classValues[i]:
                counts[i] += 1
    return classValues[counts.index(max(counts))]


def partition(set, attr):
    """
    Dataset Partition on given attribute.
    Returns dict with each subset according to attr values in data_spec
    """
    partitioned_set = {}
    for tag in data_spec[attr]:  # Split according to each value of attribute
        subset = []  # Temp var for holding subset
        index = attributes.index(attr)
        for row in set:
            if row[index].upper() == tag.upper():
                subset.append(row)  # Add row to subset
        partitioned_set[tag] = subset  # Store subset in dict
    return partitioned_set


def partition_continuous(data, attr, threshold):
    partitioned_set = {}
    partitioned_set[0] = []
    partitioned_set[1] = []
    for row in data:
        attr_index = attributes.index(attr)
        index = 0 if row[attr_index] < threshold else 1
        partitioned_set[index].append(row)
    return partitioned_set


def entropy(subset, classifier):
    """
    Calculate entropy of a set.
    Parameters are
    subset to calculate the entropy on, and
    the classifier to use
    Returns the entropy of the data set
    """
    index = -1
    entropy = 0
    total_rows = len(subset)
    for value in data_spec[classifier]:
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
            if "threshold" in path:
                decision = 0 if row[attr_index] < path["threshold"] else 1
            else:
                decision = row[attr_index]
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
        attr = attr.lower()
        index += 1
        if line.endswith("}"):
            # Case Attribute
            attributes.append(attr)
            vals = line[index:].lstrip(" { ").rstrip(" } ").split(", ")
            for index in range(len(vals)):
                vals[index] = vals[index].lower()
            data_spec[attr] = vals
        elif line[-4:] == "Real":
            # Case Continuous
            attributes.append(attr)
            data_spec[attr] = "continuous"
            pass
        else:
            # Case Class
            index = line.find(":")
            attr = line[:index]
            classes.append(attr)
            index += 1
            vals = line[index:].strip().split(" ")
            for index in range(len(vals)):
                vals[index] = vals[index].lower()
            data_spec[attr] = vals

    file.close()


def read_data(filePath):
    """
    Reads in the data from data.dat to be processed in the tree induction
    """
    file = open(filePath)
    for line in file:
        vals = line.strip().split()
        for index in range(len(vals)):
            if has_continuous and is_real(vals[index]):
                vals[index] = float(vals[index])
            else:
                vals[index] = vals[index].lower()
        training_dataset.append(vals)
    split_data()
    file.close()


def split_data():
    num_rows = len(training_dataset)
    num_test = floor(0.3*num_rows)
    for i in range(num_test):
        index = randint(0, num_rows - 1)
        row = training_dataset.pop(index)
        test_dataset.append(row)
        num_rows -= 1


################
# "Main Method"
################
if __name__ == '__main__':
    spec_path = "data/data.spec"
    data_path = "data/data.dat"
    if len(sys.argv) > 3:
        spec_path = sys.argv[2]
        data_path = sys.argv[3]

    command = sys.argv[1]
    command = command[1:]
    if command[0] == 'c':
        has_continuous = True
    elif command[0] == 'm':
        has_missing = True
    elif command[0] == 'p':
        must_prune = True

    min_test = 101
    max_test = -1
    average = 0

    read_spec(spec_path)
    read_data(data_path)
    if has_missing or must_prune:
        print(("Command {s} not yet implemented").format(s=command))
    elif has_continuous:
        tree = train_continuous(training_dataset, attributes)
        print_continuous(tree, classes[0])
    else:
        tree = train_discrete(training_dataset, attributes)
        print_discrete(tree, classes[0])
    error = classify(tree, test_dataset)
    average += error
    if error > max_test:
        max_test = error
    if error < min_test:
        min_test = error
    split_data()
