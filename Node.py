class Node:
    label = "" # Represents decision to be made/classification of leaf
    attrValue = "" # Value of attribute from parent decision (label)
    children = [] # Should be of type Node
    isLeaf = False # Whether Node is a leaf node or not

    def __init__(self, label = "", leaf = False):
        self.label = label
        self.isLeaf = leaf
    
    def add_child(self, child):
        self.children.append(child)