####################
# Import Statements
####################
import math
import Node
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
def trainDiscrete():
    pass

# Dataset Partition on given attribute.
# Returns dict with each subset according to attr values in dataSpec
def partition(set, attr):
    partitionedSet = {}
    for tag in dataSpec[attr]: # Split according to each value of attribute
        subset = [] # Temp var for holding subset
        index = attributes.index(attr)
        for row in dataset:
            if row[index].upper() == tag.upper():
                subset.append(row) # Add row to subset
        partitionedSet[tag] = subset # Store subset in dict
    return partitionedSet

# Calculate entropy of a set.
# Parameters are the 
#   subset to calculate the entropy on, and 
#   the classTag to use
# Returns the entropy of the data set
def entropy(subset, classTag):
    # Calculate index for class field. Usually -1. But can accomodate multiple classes
    index = - len(classes) - classes.index(classTag)
    entropy = 0
    totalRows = len(subset)
    for value in dataSpec[classTag]:
        count = 0
        for row in subset:
            if row[index] == value:
                count += 1
        prob = float(count) / totalRows
        if prob == 0:
            entropy += 0
        else:
            entropy += prob * math.log(prob, 2)
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

    print(dataset)