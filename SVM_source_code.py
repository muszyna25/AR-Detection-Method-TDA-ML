    <Support Vector Machine classifer for classifcation of atmospheric rivers based on topological feature descriptors.
    This work was supported by Intel Parallel Computing Center at University of Liverpool, UK and funded by Intel.>
    Copyright (C) <2018>  <Grzegorz Muszynski>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from sklearn.externals import joblib
import os
import collections
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import LabelKFold
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.grid_search import GridSearchCV

n_jobs_ = 480

os.chdir('../')

X_test = np.loadtxt('CAM100_gridsearch_data.txt')
y_test = np.loadtxt('CAM100_gridsearch_labels.txt')

X_train = np.loadtxt('CAM100_cross_val_data.txt')
y_train = np.loadtxt('CAM100_cross_val_labels.txt')

scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)
X_train = scaler.fit_transform(X_train)

cv_1 = StratifiedKFold(y_test, n_folds=10, random_state=12)

C_range = np.logspace(-5, 15, 11, base=2)
gamma_range = np.logspace(-15, 3, 10, base=2)
param_grid = dict(gamma=gamma_range, C=C_range)
    
#Find the best parameters for SVM.
grid = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=cv_1, scoring='accuracy', n_jobs = 320, verbose=2)
print("GridSearchCV line... done")
grid.fit(X_test, y_test)
print("grid.fit() line... done")

print(grid.get_params())

print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

print(grid.best_estimator_)
clf = grid.best_estimator_ 

cv_2 = StratifiedKFold(y_train, n_folds=10, random_state=12)

scores = cross_val_score(clf, X_train, y_train, cv=cv_2, scoring='accuracy', n_jobs=n_jobs_)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

predicted_test_cross = cross_val_predict(clf, X_train, y_train, cv=cv_2, n_jobs=n_jobs_)
print(np.count_nonzero(predicted_test_cross == 1, axis=0))

cmat = confusion_matrix(y_train, predicted_test_cross, labels=[0,1])
print("Confusion matrix.:", cmat)

#Precision = true positives / (true positives + false positives)
precision_score_cross_val = metrics.precision_score(y_train, predicted_test_cross, pos_label=1, average='binary')
print("Precision score cross val.: %0.2f" % precision_score_cross_val)

#Recall = true positives / (true positives + false negatives)
recall_score_cross_val = metrics.recall_score(y_train, predicted_test_cross, pos_label=1, average='binary')
print("Recall score cross val.: %0.2f" % recall_score_cross_val)

#F1 score =  2 * (precision * recall) / (precision + recall)
f1_score_cross_val = metrics.f1_score(y_train, predicted_test_cross, pos_label=1, average='binary')
print("F1 score cross val.: %0.2f" % (f1_score_cross_val))
