# Classification Decision Tree

My implementation of an algorithm to induce a classification decision tree, written in python3.

All the code is included in the DecisionTree.py.

This program is run as follows:

`python3 DecisionTree.py <option> data.spec data.dat`

where option can be one of four options:
- `-d`: Induce a tree only with discrete data
- `-c`: Induce a tree that contains continuous values
- `-md`: Induce a tree, in addition to discrete data, may contain missing values
- `-pd`: Induce a tree, in addition to discrete data, prune the tree after induction with a limit of 5% generalisation error deterioration

Additional data files and their spec files can be found in the data folder