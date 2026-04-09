# Description

Repository containing the code of the paper entitled “A state reduction approach for learning-based model predictive control for train rescheduling” published in IFAC-PapersOnLine, available [here](https://www.sciencedirect.com/science/article/pii/S2405896325027417).

-------------------

# Files

analysis_test_single.ipynb: analyses the results obtained from "tests_learning_cl_single.py".

analysis_tests_learning_all.ipynb: analyses the results obtained from "tests_learning_cl_ensemble_all.py". The results from the paper's table come from here.

gurobi.lic: GUROBI solver license file.

rail_data_preprocess_reduced.py: defines functions for data pre-processing.

rail_fun.py: creates auxiliary functions for data pre-processing, computing the step cost, and retrieving information from the system.

rail_gen_optimal_data_reduced.py: generates the data used for the training of the supervised learning approach by solving the underlying mixed-integer nonlinear (or linear) program. The flag "testing=False" sets the script to be run in a computing cluster. Alternatively, "testing=True" is used for local small tests.

rail_learning_cluster_reduced.py: trains the neural networks and saves the resulting weights and hyperparameters. The flag "testing=False" sets the script to be run in a computing cluster. Alternatively, "testing=True" is used for local small tests.

rail_training_reduced.py: creates classes which represent several neural network architectures, creates a function which updates the neural network weights by backpropagation, and defines auxiliary functions for data pre-processing and testing.

tests_learning_cl_ensemble.py: tests the closed-loop performance of the ensemble of the selected neural networks. This script only tests the MILP and Learning+LP methods.

tests_learning_cl_ensemble_all.py: similarly as above, it tests the closed-loop performance of the ensemble of the selected neural networks. This script is more complete and it also tests the MINLP(10minutes), MINLP(with warm-start) and Learning+NLP approaches.

tests_learning_cl_single.py: individually tests the closed-loop performance of the trained neural networks.

tests_ntbk2.ipynb: notebook with several tests and experiments.
