####################
# Import Statements
####################
#from math import log
import math
import operator

########################
# Variable declarations
########################
dataSpec = {} # Dict for holding all dataSpec as keys and a list of their values
attributes = [] # list of attributes in dataSpec
classes = [] # list of the classes in dataSpec
dataset = [] # list of lists containing all the data to be processed for the tree induction

########################
# Function Declarations
########################
# Dataset Partition on given attribute
def partition(set, attr):
    pass

# Calculate entropy of a set
def entropy(subset, classTag):
    # Calculate index for class field. Usually -1
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
def readSpec():
    file = open("bin/data/data.spec")
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
def readData():
    file = open("bin/data/data.dat")
    for line in file:
        dataset.append(line.strip().split(' '))
    file.close()


################
# "Main Method"
################
if __name__ == '__main__':
    readSpec()

    print(dataSpec)
    print(attributes)
    print(classes)

    readData()

    # for row in dataset:
        # print(row)
    entropy = entropy(dataset, classes[0])
    print(entropy)