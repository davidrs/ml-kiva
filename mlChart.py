import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd

data = pd.read_csv("output/train.csv")
data_test = pd.read_csv("output/test.csv")
#print data

X = data[['loan_amount', 'gender', 'pictured']].as_matrix() # select columns
X_test = data_test[['loan_amount', 'gender', 'pictured']].as_matrix() # select columns

# scale to <0,1>
max_all = np.amax(X, 0)

X = 1.0 * X / max_all
X_test = 1.0 * X_test / max_all
Y = data.expired.as_matrix()   # select column
Y_test = data.expired.as_matrix()   # select column

# figure number
fignum = 1

print X
print Y


# fit the model
for name, penalty in (('unreg', 1), ('reg', 0.05)):
    print "iterate" + str(name)
    clf = svm.SVC(kernel='linear', C=penalty)
    clf.fit(X, Y)
    print "done fit"

    # get the separating hyperplane
    w = clf.coef_[0]
    print w
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy + a * margin
    yy_up = yy - a * margin

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(6, 4))
    plt.clf()
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10)
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired)

    plt.axis('tight')
    x_min = -4.8
    x_max = 4.2
    y_min = -6
    y_max = 6

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    ##print XX.ravel()
    print "---dd-d--d-d"
    ##print np.c_[XX.ravel(), YY.ravel()]
    #Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])
    Z = clf.predict(X_test)
    print Z
    # Put the result into a color plot
    Z = Z.reshape(X_test.shape)
    plt.figure(fignum, figsize=(6, 4))
    plt.pcolormesh(X_test, Y_test, Z, cmap=plt.cm.Paired)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    fignum = fignum + 1

plt.show()
quit