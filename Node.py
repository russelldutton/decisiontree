class Node:
    label = "" # Represents decision to be made/classification of leaf
    attrValue = "" # Value of attribute from parent decision (label)
    children = [] # Should be of type Node
    dataset = [] # Will be a list of lists
    isLeaf = False # Whether Node is a leaf node or not

    def __init__(self, subset, attrValue):
        self.dataset = subset
        self.attrValue = attrValue
        self.isLeaf = False
    
    
