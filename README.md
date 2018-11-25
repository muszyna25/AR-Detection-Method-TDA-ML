# AR-Detection-Method-TDA-ML
Atmospheric River Pattern Detection Method (Topological Data Analysis + Machine Learning)

Paper: "Topological Data Analysis and Machine Learning for Recognizing Atmospheric River Patterns in Large Climate Datasets"

Grzegorz Muszynski (1,2), Karthik Kashinath (2), Vitaliy Kurlin (1), Michael Wehner (2), and Prabhat (2)

1) Department of Computer Science, University of Liverpool, Liverpool, L69 3BX, United Kingdom
2) Lawrence Berkeley National Laboratory, Berkeley, California, 94720, United States

-------------------------------------------------------------------------------------------------------------------------------

1) TDA_source_code.cxx contains a code for computing topological feature descriptors (connected components) from snapshots of climate simulations in a threshold-free way. This code works only with TECA software (https://github.com/LBL-EESA/TECA). TECA includes all necessary external packages and libraries for this implementation.

2) SVM_source_code.py contains all code to train and test Support Vector Machine classifier (implemented in Python scikit-learn).


