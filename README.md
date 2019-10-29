# AR-Detection-Method-TDA-ML
A Working Prototype of Atmospheric River Pattern Detection Method (Topological Data Analysis + Machine Learning)

Paper: Muszynski, Grzegorz, et al. "Topological data analysis and machine learning for recognizing atmospheric river patterns in large climate datasets." Geoscientific Model Development 12.2 (2019): 613-628.

Grzegorz Muszynski (1,2), Karthik Kashinath (2), Vitaliy Kurlin (1), Michael Wehner (2), and Prabhat (2)

1) Department of Computer Science, University of Liverpool, Liverpool, L69 3BX, United Kingdom
2) Lawrence Berkeley National Laboratory, Berkeley, California, 94720, United States

-------------------------------------------------------------------------------------------------------------------------------

1) TDA_source_code.cxx contains a C++ code for computing topological feature descriptors (connected components) from snapshots of climate simulations in a threshold-free way. This code works only with TECA software (https://github.com/LBL-EESA/TECA). TECA includes all necessary external packages and libraries for this implementation.

2) Preprocessing_ouput_files.py preprocesses output files of topological feature extraction in 1).

2) SVM_source_code.py contains all code to train and test Support Vector Machine classifier implemented in Python scikit-learn  (https://scikit-learn.org/stable/modules/svm.html). It requires an installion of python3.5/3.6 and all necessary modules.


