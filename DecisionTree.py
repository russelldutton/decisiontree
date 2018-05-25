####################
# Import Statements
####################
from math import log, floor
from random import randint
import copy
import sys


########################
# Variable declarations
########################
# Dict for holding all data_spec as keys and a list of their values
data_spec = {}
# list of attributes in data_spec
attributes = []
# class of the data spec
class_name = ""
# list of lists containing all the data to be processed for the tree induction
training_dataset = []
# data to test tree classification
test_dataset = []
# condition to account for missing values
has_missing = False
# condition to account for continuous data
has_continuous = False
# condition whether the tree should be pruned after being created or not
must_prune = False
# List of rules for the tree
rules = []
# Dict holding counts of total rows correctly classified
total_correct = {"test": 0, "train": 0}

########################
# Function Declarations
########################


def train_continuous(subset):
    """
    Function that recursively trains the tree using continuous values.
    Takes the subset to train with for this call as a parameter.

    Similar to train_discrete, I implemented this function separately
    for my own sanity
    """
    data = subset[:]
    default = get_default(data)
    node = {}

    if len(data) == 0:
        node["label"] = default
        node["is_leaf"] = True
        node["patterns"] = 0
        node["test"] = 0
        node["num_test"] = 0
        node["train"] = 0
        node["num_train"] = 0
        return node
    elif is_homogenous(data, class_name):
        node["label"] = data[0][-1]
        node["is_leaf"] = True
        node["patterns"] = len(data)
        node["test"] = 0
        node["num_test"] = 0
        node["train"] = 0
        node["num_train"] = 0
        return node
    else:
        (best_attr, best_sets,
         threshold) = get_best_attribute(attributes, class_name, data)
        node["label"] = best_attr
        node["is_leaf"] = False
        node["children"] = {}
        if threshold is None:
            # attrs.remove(best_attr)
            for o in best_sets:
                child = train_continuous(best_sets[o])
                node["children"][o] = child
        else:
            node["threshold"] = threshold
            # attrs.remove(best_attr)
            for o in best_sets:
                child = train_continuous(best_sets[o])
                node["children"][o] = child
        return node


def train_discrete(subset):
    """
    Discrete data algorithm.
    Induces the tree recursively.
    Takes the subset to induce for this instance of the function
    """
    data = subset[:]
    default = get_default(data)

    node = {}

    if len(data) == 0:   # If empty dataset
        node["label"] = default
        node["is_leaf"] = True
        node["patterns"] = 0
        node["test"] = 0
        node["num_test"] = 0
        node["train"] = 0
        node["num_train"] = 0
        return node
    elif is_homogenous(data, class_name):  # Homogenous dataset
        node["label"] = data[0][-1]
        node["is_leaf"] = True
        node["patterns"] = len(data)
        node["test"] = 0
        node["num_test"] = 0
        node["train"] = 0
        node["num_train"] = 0
        return node
    else:  # Heterogenous dataset
        (best_attr, best_sets,
         _t) = get_best_attribute(attributes, class_name, data)
        node = {"label": best_attr, "is_leaf": False, "children": {}}
        for o in best_sets:
            if len(best_sets[o]) == len(data):
                node["label"] = data[0][-1]
                node["is_leaf"] = True
                node["patterns"] = len(data)
                node["test"] = 0
                node["num_test"] = 0
                node["train"] = 0
                node["num_train"] = 0
                return node
            child = train_discrete(best_sets[o])
            node["children"][o] = child
        return node


def discrete_to_rules(tree, rule_string=""):
    """
    Recursively add rules to the rules list for the tree produced
    by the train_discrete function.
    takes the tree to print, and the rule to be printed
    """
    if tree["is_leaf"] is True:
        rule_string += "THEN {s1} IS {s2}"
        rule = {"rule": rule_string.format(s1=class_name, s2=tree["label"])}
        rule["patterns"] = tree["patterns"]
        if tree["num_test"] != 0:
            rule["test"] = float(tree["test"])/float(tree["num_test"])
            rule["test"] = round(1 - rule["test"], 2)
        else:
            rule["test"] = 0
        if tree["num_train"] != 0:
            rule["train"] = float(tree["train"])/float(tree["num_train"])
            rule["train"] = round(1 - rule["train"], 2)
        else:
            rule["train"] = 0
        rules.append(rule)
        tree["test"] = 0
        tree["num_test"] = 0
        tree["train"] = 0
        tree["num_train"] = 0
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
            discrete_to_rules(tree["children"][key], out)


def continuous_to_rules(tree, rule_string=""):
    """
    Recursively add rules to the rules list for the tree produced
    by the train_continuous function.
    takes the tree to print, and the rule to be printed
    """
    if tree["is_leaf"]:
        rule_string += "THEN {s1} IS {s2}"
        rule = {"rule": rule_string.format(s1=class_name, s2=tree["label"])}
        rule["patterns"] = tree["patterns"]
        if tree["num_test"] != 0:
            rule["test"] = float(tree["test"])/float(tree["num_test"])
            rule["test"] = round(1 - rule["test"], 2)
        else:
            rule["test"] = 0
        if tree["num_train"] != 0:
            rule["train"] = float(tree["train"])/float(tree["num_train"])
            rule["train"] = round(1 - rule["train"], 2)
        else:
            rule["train"] = 0
        rules.append(rule)
        tree["test"] = 0
        tree["num_test"] = 0
        tree["train"] = 0
        tree["num_train"] = 0
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
            continuous_to_rules(tree["children"][key], out)


def print_rules():
    out = ""
    for rule in rules:
        out += "["
        out += str(rule["patterns"])
        out += "  "
        out += str(rule["train"])
        out += "  "
        out += str(rule["test"])
        out += "]\t"
        out += rule["rule"]
        out += "\n"
    print(out)


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
    best_gain = -1
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
        if _gain > best_gain or best_gain < 0:
            best_attr = x
            best_sets = _sets
            best_gain = _gain
            best_threshold = _threshold
    return best_attr, best_sets, best_threshold


def get_best_discrete(data, attribute):
    """
    Get the best attribute to split on for discrete data after splitting
    on the attribute provided in the parameter list
    """
    data_entropy = entropy(data, class_name)
    subsets = partition(data, attribute)
    split_entropy = 0
    split_info = 0
    for o in subsets:
        outcome_probability = float(len(subsets[o]))/float(len(data))
        subset_entropy = entropy(subsets[o], class_name)
        subset_entropy *= outcome_probability
        split_entropy += subset_entropy
        temp = 0
        if outcome_probability > 0:
            temp += outcome_probability
            temp *= log(outcome_probability, 2)
        split_info += temp
    gain = data_entropy - split_entropy
    if has_missing:
        prob = get_prob_attr_known(data, attribute)
        prob /= len(data)
        gain = (1 - prob) * gain
    elif split_info != 0:
        gain /= -1.0 * split_info
    return subsets, gain


def get_prob_attr_known(data, attribute):
    """
    Get the probability an attribute is known, used for missing values
    """
    count = 0
    index = attributes.index(attribute)
    for row in data:
        if row[index] == '?':
            count += 1
    return count


def get_best_continuous(data, attribute):
    """
    Get the best attribute to split on after having split on the attribute
    provided in the parameter list
    """
    data_entropy = entropy(data, class_name)
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
            subset_entropy = entropy(subsets[o], class_name)
            subset_entropy *= outcome_probability
            split_entropy += subset_entropy
            temp = 0
            if outcome_probability > 0:
                temp += outcome_probability
                temp *= log(outcome_probability, 2)
            split_info += temp
        gain = data_entropy - split_entropy
        if gain > best_gain:
            best_threshold = threshold
            best_gain = gain
            best_sets = subsets
    return best_sets, best_gain, best_threshold


def get_unique_list(data, row_index):
    """
    Return a list of unique values for the attribute at
    row_index, in the data subset
    """
    values = []
    for row in data:
        if row[row_index] not in values:
            values.append(float(row[row_index]))
    values.sort()
    return values


def is_real(val):
    """
    Returns True if value is a Real number, False otherwise
    """
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False


def get_default(subset):
    """
    Get the majority class of the subset
    """
    counts = []
    classValues = data_spec[class_name]
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

    Note: In the case of missing values, I have opted to exclude them from
    any partitioning
    """
    partitioned_set = {}
    for tag in data_spec[attr]:  # Split according to each value of attribute
        subset = []  # Temp var for holding subset
        index = attributes.index(attr)
        for row in set:
            if row[index] == tag:
                subset.append(row)  # Add row to subset
        partitioned_set[tag] = subset  # Store subset in dict
    return partitioned_set


def partition_continuous(data, attr, threshold):
    """
    Partition a data set of continuous values, on the threshold provided
    """
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


def contains_missing(row):
    """
    Returns true if the row contains a missing value somewhere
    """
    for e in row:
        if e == '?':
            return True
    return False


def get_num_patterns(tree, row, patterns):
    """
    This function recursively counts the number of patterns associated with
    each of the values for the class of the dataset, and returns them. It
    travels the tree and returns the counts of patterns that the row with
    a missing value could possibly have belonged to. The result of this
    function is used to classify said row.

    It is used to classify rows with missing values. In the case of a missing
    value, the function will count the patterns for all the sub-branches,
    but will continue to only count relevant branches to the row
    """
    if tree["is_leaf"] is True:
        patterns[tree["label"]] += tree["patterns"]
        return patterns
    else:
        attr_index = attributes.index(tree["label"])
        if row[attr_index] != "?":
            decision = row[attr_index]
            tree = tree["children"][decision]
            patterns = get_num_patterns(tree, row, patterns)
            return patterns
        else:
            for c in tree["children"]:
                path = tree["children"][c]
                get_num_patterns(path, row, patterns)
            return patterns


def find_max_patterns(patterns):
    """
    This function simply returns the class value with the highest count
    of patterns, and is used in conjunction with get_num_patterns
    """
    max_pattern = None
    max_value = -1
    for p in patterns:
        if max_pattern is None or max_value < patterns[p]:
            max_pattern = p
            max_value = patterns[p]
    return max_pattern


def classify(tree, dataset, test_type="test"):
    """
    This function classifies each row in the dataset using the tree.

    It also counts number of successful classifications and total
    number of classifications for either the test or training dataset,
    with the test dataset being the default
    """
    correct = 0
    for row in dataset:
        path = tree
        if has_missing and contains_missing(row):
            patterns = {}
            for c in data_spec[class_name]:
                patterns[c] = 0
            get_num_patterns(path, row, patterns)
            max_class = find_max_patterns(patterns)
            if row[-1] == max_class:
                correct += 1
                total_correct[test_type] += 1
        else:
            before_traverse = total_correct[test_type]
            tree = traverse_tree(path, row, test_type)
            correct += total_correct[test_type] - before_traverse
    return correct


def traverse_tree(tree, row, test_type="test"):
    if tree["is_leaf"] is True:
        if row[-1] == tree["label"]:
            tree[test_type] += 1
            total_correct[test_type] += 1
        tree["num_" + test_type] += 1
        return tree
    else:
        attr_index = attributes.index(tree["label"])
        if "threshold" in tree:
            decision = 0 if row[attr_index] < tree["threshold"] else 1
        else:
            decision = row[attr_index]
        tree["children"][decision] = traverse_tree(tree["children"][decision],
                                                   row, test_type)
        return tree


def is_deep_decision(tree):
    """
    This function decides whether the root node of the tree only has
    leaves as children to prune
    """
    if tree["is_leaf"] is True:
        return False
    else:
        for c in tree["children"]:
            if tree["children"][c]["is_leaf"] is False:
                return False
        return True


def get_majority(tree):
    """
    This function gets the majority class of the leaves of the root node
    of the tree
    """
    class_proportions = {}
    for e in data_spec[class_name]:
        class_proportions[e] = 0
    max_prop = -1
    max_class = ""
    for c in tree["children"]:
        child = tree["children"][c]
        class_proportions[child["label"]] += int(child["patterns"])

    for p in class_proportions:
        if class_proportions[p] > max_prop:
            max_prop = class_proportions[p]
            max_class = p
    return max_class


def prune(tree, full_tree, initial_error):
    """
    Function to prune the tree.

    Tree is a subtree of full_tree and is a reference to a reference
    to the subtree in full_tree. So any changes made to tree reflect
    in full_tree
    """
    path = tree
    if is_deep_decision(tree):
        path = copy.deepcopy(tree)
        default = get_majority(path)
        patterns = 0
        for c in path["children"]:
            patterns += path["children"][c]["patterns"]
        del path["children"]
        path["is_leaf"] = True
        path["label"] = default
        path["patterns"] = patterns
        path["num_test"] = 0
        path["num_train"] = 0
        path["test"] = 0
        path["train"] = 0
        test_error_after = classify(full_tree, test_dataset)
        if initial_error - test_error_after < 5:
            initial_error -= test_error_after
            del tree
            tree = copy.deepcopy(path)
            del path
        return tree
    else:
        if path["is_leaf"] is True:
            return tree
        else:
            for c in path["children"]:
                tree["children"][c] = prune(path["children"][c],
                                            full_tree, initial_error)
        return tree


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
            global class_name
            index = line.find(":")
            attr = line[:index]
            class_name = attr
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
    """
    Splits the data into 70% training data, 30% test data
    """
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
    spec_path = "data.spec"
    data_path = "data.dat"
    if len(sys.argv) > 3:
        spec_path = sys.argv[2]
        data_path = sys.argv[3]

    command = sys.argv[1]
    command = command[1:]
    if command == "c":
        has_continuous = True
    elif command == "md":
        has_missing = True
    elif command == "pd":
        must_prune = True
    elif command != "d":
        print("Command not recognized. Valid options are\n-c\n-md\n-pd")
        exit()

    read_spec(spec_path)
    read_data(data_path)
    if has_continuous:
        print("*********************")
        print("  CONTINUOUS DATA")
        print("*********************")
        tree = train_continuous(training_dataset)
        classify(tree, training_dataset, "train")
        classify(tree, test_dataset, "test")
        continuous_to_rules(tree)
        for c in total_correct:
            num_rows = 0
            title = ""
            if c == "test":
                num_rows = len(test_dataset)
                title = "Testing Classification Error: "
            else:
                num_rows = len(training_dataset)
                title = "Training Classification Error: "
            error = 100 - round(total_correct[c] / num_rows, 2) * 100
            print(title, error, "%")
        print_rules()
    else:
        print("*********************")
        if must_prune:
            print(" DISCRETE - BEFORE PRUNING")
        elif has_missing:
            print(" DISCRETE WITH MISSING")
        else:
            print(" DISCRETE DATA")
        print("*********************\n")
        tree = train_discrete(training_dataset)
        classify(tree, training_dataset, "train")
        classify(tree, test_dataset, "test")
        discrete_to_rules(tree)

        print(total_correct)

        for c in total_correct:
            num_rows = 0
            title = ""
            if c == "test":
                num_rows = len(test_dataset)
                title = "Testing Classification Error: "
            else:
                num_rows = len(training_dataset)
                title = "Training Classification Error: "
            error = 100 - round(total_correct[c] / num_rows, 2) * 100
            print(title, error, "%")
        print_rules()
        print()
        if must_prune:
            rules = []
            total_correct["test"] = 0
            total_correct["train"] = 0
            test_error_before = total_correct["test"]/len(test_dataset)
            test_error_before = round(test_error_before, 2) * 100
            tree = prune(tree, tree, test_error_before)
            classify(tree, training_dataset, "train")
            classify(tree, test_dataset, "test")
            discrete_to_rules(tree)

            print(total_correct)

            for t in total_correct:
                num_rows = 0
                if t == "test":
                    num_rows = len(test_dataset)
                    title = "Testing Classification Error: "
                else:
                    num_rows = len(training_dataset)
                    title = "Training Classification Error: "
                error = 100 - round(total_correct[t] / num_rows, 2) * 100
                print(title, error, "%")
            print_rules()
