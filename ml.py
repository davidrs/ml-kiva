"""
vectorize categorical variables
optionally train an SVM and a random forest, get validation AUC

importing from another script:
from vectorize_validation import y_train, x_train, y_test, x_test
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

import numpy as np
import pandas as pd
import sqlite3

from math import sqrt
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC

###

data_dir = 'output/'	# needs trailing slash

# validation split, both files with headers and the Expired column
train_file = data_dir + 'train.csv'
test_file = data_dir + 'test.csv'

###
train = pd.read_csv( train_file )
test = pd.read_csv( test_file )


# numeric x
numeric_cols = [ 'loan_amount' ]
x_num_train = train[ numeric_cols ].as_matrix()
x_num_test = test[ numeric_cols ].as_matrix()

# scale to <0,1>
max_train = np.amax( x_num_train, 0 )
max_test = np.amax( x_num_test, 0 )		# not really needed

x_num_train = 1.0 * x_num_train / max_train
x_num_test = 1.0 * x_num_test / max_train		# scale test by max_train
print "x_num_train"
print x_num_train

# y
y_train = train.expired
y_test = test.expired

# categorical
cat_train = train.drop( numeric_cols , axis = 1 )
cat_test = test.drop( numeric_cols , axis = 1 )

cat_train.fillna( 'NA', inplace = True )
cat_test.fillna( 'NA', inplace = True )

x_cat_train = cat_train.T.to_dict().values()
x_cat_test = cat_test.T.to_dict().values()

# vectorize

vectorizer = DV( sparse = False )
vec_x_cat_train = vectorizer.fit_transform( x_cat_train )
vec_x_cat_test = vectorizer.transform( x_cat_test )

# complete x

x_train = np.hstack(( x_num_train, vec_x_cat_train ))
x_test = np.hstack(( x_num_test, vec_x_cat_test ))


def drawChart():
	# figure number
	fignum = 1

	# fit the model
	for name, penalty in (('unreg', 1), ('reg', 0.05)):

	    clf = svm.SVC(kernel='linear', C=penalty)
	    clf.fit(x_train, y_train)

	    # get the separating hyperplane
	    w = clf.coef_[0]
	    a = -w[0] / w[1]
	    xx = np.linspace(-5, 5)
	    yy = a * xx - (clf.intercept_[0]) / w[1]

	    # plot the parallels to the separating hyperplane that pass through the
	    # support vectors
	    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
	    yy_down = yy + a * margin
	    yy_up = yy - a * margin

	    # plot the line, the points, and the nearest vectors to the plane
	    plt.figure(fignum, figsize=(4, 3))
	    plt.clf()
	    plt.plot(xx, yy, 'k-')
	    plt.plot(xx, yy_down, 'k--')
	    plt.plot(xx, yy_up, 'k--')

	    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
	                facecolors='none', zorder=10)
	    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, zorder=10, cmap=plt.cm.Paired)

	    plt.axis('tight')
	    x_min = -2
	    x_max = 2
	    y_min = -2
	    y_max = 2

	    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
	    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

	    # Put the result into a color plot
	    Z = Z.reshape(XX.shape)
	    plt.figure(fignum, figsize=(4, 3))
	    plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

	    plt.xlim(x_min, x_max)
	    plt.ylim(y_min, y_max)

	    plt.xticks(())
	    plt.yticks(())
	    fignum = fignum + 1

	plt.show()




def trainModels():
	# SVM looks much better in validation

	print "training SVM..."
	
	# although one needs to choose these hyperparams
	C = 20 #173
	gamma =  0.001 #1.31e-5
	shrinking = True

	probability = True
	verbose = True

	svc = SVC( C = C, 
		gamma = gamma, 
		shrinking = shrinking, 
		probability = probability, 
		verbose = verbose )
	svc.fit( x_train, y_train )
	p = svc.predict_proba( x_test )	
	
	print x_test[12]
	print svc.predict_proba( x_test[12] )
	print svc.predict_proba( x_test[13] )
	print svc.predict_proba( x_test[14] )


	auc = AUC( y_test, p[:,1] )
	print "SVM AUC", auc	
	

	print "training random forest..."

	n_trees = 100
	max_features = int( round( sqrt( x_train.shape[1] ) * 2 ))		# try more features at each split
	max_features = 'auto'
	verbose = 1
	n_jobs = 1

	rf = RF( n_estimators = n_trees, 
		max_features = max_features, 
		verbose = verbose, 
		n_jobs = n_jobs )
	rf.fit( x_train, y_train )

	p = rf.predict_proba( x_test )

	
	print x_test[12]
	print rf.predict_proba( x_test[12] )
	print rf.predict_proba( x_test[13] )
	print rf.predict_proba( x_test[14] )

	auc = AUC( y_test, p[:,1] )
	print "RF AUC", auc

	# AUC 0.701579086548
	# AUC 0.676126704696

	# max_features * 2
	# AUC 0.710060065732
	# AUC 0.706282346719


if __name__ == "__main__":
	drawChart()
	#trainModels()
