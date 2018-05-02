####################
# Import Statements
####################
from math import log
from Node import Node
import sys

########################
# Variable declarations
########################
dataSpec = {} # Dict for holding all dataSpec as keys and a list of their values
attributes = [] # list of attributes in dataSpec
classes = [] # list of the classes in dataSpec
dataset = [] # list of lists containing all the data to be processed for the tree induction
root = None

########################
# Function Declarations
########################
# Discrete data algorithm
def trainDiscrete(subset, attribute_list):
    data = subset[:]
    attrs = attribute_list[:]
    default = getDefault(subset)
    data_entropy = entropy(data, classes[0])

    if len(data) == 0: # If empty dataset
        node = Node(default, True)
        return node
    elif isHomogenous(data, classes[0]): # Homogenous dataset
        node = Node(data[0][-1], True)
        return node
    else: # Heterogenous dataset
        best_gain = 0
        best_sets = None
        best_attr = None
        for x in attrs:
            subsets = partition(data, x)
            split_entropy = 0
            split_info = 0
            for o in subsets:
                outcome_probability = float(len(subsets[o]))/float(len(data))
                split_entropy += entropy(subsets[o], classes[0]) * outcome_probability
                split_info += outcome_probability * log(outcome_probability, 2)
            gain = data_entropy - split_entropy
            gain /= (-1) * split_info
            if gain > best_gain:
                best_gain = gain
                best_sets = subsets
                best_attr = x
        node = {best_attr: {}}
        # return node
        attrs.remove(best_attr)
        for o in best_sets:
            child = trainDiscrete(best_sets[o], attrs)
            node[best_attr] = Node(best_attr, False).add_child(o, child)
    return node


def isHomogenous(data, classifier):
    tag = data[0][-1]
    for row in data:
        if tag != row[-1]:
            return False
    return True

def getBestAttribute(attrs, class_val, subset):
    pass

def getDefault(subset):
    counts = []
    classValues = dataSpec[classes[0]]
    for i in range(len(classValues)):
        counts.append(0)
        for row in subset:
            if row[-1] == classValues[i]:
                counts[i] += 1 
    return classValues[counts.index(max(counts))]

# Dataset Partition on given attribute.
# Returns dict with each subset according to attr values in dataSpec
def partition(set, attr):
    partitionedSet = {}
    for tag in dataSpec[attr]: # Split according to each value of attribute
        subset = [] # Temp var for holding subset
        index = attributes.index(attr)
        for row in set:
            if row[index].upper() == tag.upper():
                subset.append(row) # Add row to subset
        partitionedSet[tag] = subset # Store subset in dict
    return partitionedSet

# Calculate entropy of a set.
# Parameters are the 
#   subset to calculate the entropy on, and 
#   the classifier to use
# Returns the entropy of the data set
def entropy(subset, classifier):
    # Calculate index for class field. Usually -1. But can accomodate multiple classes
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


# Reads in and interprets the data.spec file that details the data properties for the algorithm
def readSpec(filePath):
    file = open(filePath)
    #spec = file.read()

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

# Reads in the data from data.dat to be processed in the tree induction
def readData(filePath):
    file = open(filePath)
    for line in file:
        dataset.append(line.strip().split(' '))
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
    readSpec(specPath)
    readData(dataPath)

    # print(dataset)
    # print(getDefault(dataset))
    tree = trainDiscrete(dataset, attributes)
    print(tree)
