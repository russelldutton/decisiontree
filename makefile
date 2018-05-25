run:
	python3 DecisionTree.py -d data.spec data.dat

run_discrete:
	python3 DecisionTree.py -d data.spec data.dat

run_continuous:
	python3 DecisionTree.py -c data.spec data.dat

run_missing:
	python3 DecisionTree.py -md data.spec data.dat

run_pruning:
	python3 DecisionTree.py -pd data.spec data.dat

