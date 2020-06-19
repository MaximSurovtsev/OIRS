# -----------------------------------
# TRAINING OUR MODEL
# -----------------------------------
import h5py
import numpy as np
import os
import glob
import cv2
import warnings
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from features import fd_histogram, fd_hu_moments, fd_haralick, fd_Fast, fd_kaze, lin

fixed_size = tuple((500, 500))

warnings.filterwarnings('ignore')

# --------------------
# tunable-parameters
# --------------------
num_trees = 100
test_size = 0.10
seed = 9
train_path = "data/train"
test_path = "data/test"
result_path = "data/result"
h5_data = 'output/data.h5'
h5_labels = 'output/labels.h5'
scoring = "accuracy"

# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()
train_labels = train_labels[1:]

if not os.path.exists(test_path):
    os.makedirs(test_path)

if not os.path.exists(result_path):
    os.makedirs(result_path)

# create all the machine learning models
models = [('LR', LogisticRegression(random_state=seed)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=seed)),
          ('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)),
          ('NB', GaussianNB())]

# variables to hold the results and names
results = []
names = []

# import the feature vector and trained labels
h5f_data = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of the feature vector and labels
print("[STATUS] features shape: {}".format(global_features.shape))
print("[STATUS] labels shape: {}".format(global_labels.shape))

print("[STATUS] training started...")

# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)

print("[STATUS] splitted train and test data...")
print("Train data  : {}".format(trainDataGlobal.shape))
print("Test data   : {}".format(testDataGlobal.shape))
print("Train labels: {}".format(trainLabelsGlobal.shape))
print("Test labels : {}".format(testLabelsGlobal.shape))

# 10-fold cross validation
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# -----------------------------------
# TESTING OUR MODEL
# -----------------------------------

# to visualize results
import matplotlib.pyplot as plt

# create the model - Random Forests
clf = RandomForestClassifier(n_estimators=num_trees, random_state=seed)

# fit the training data to the model
clf.fit(trainDataGlobal, trainLabelsGlobal)
# [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]
# loop through the test images
for idx, file in enumerate(glob.glob(test_path + "/*.jpg")):
    print(file)
    # read the image
    image = cv2.imread(file)

    # resize the image
    image = cv2.resize(image, fixed_size)

    ####################################
    # Global Feature extraction
    ####################################
    # fv_hu_moments = fd_hu_moments(image)
    # fv_haralick = fd_haralick(image)
    fv_histogram = fd_histogram(image)
    # fd_fast = fd_Fast(image)
    fv_kaze = fd_kaze(image)
    ###################################
    # Concatenate global features
    ###################################
    global_feature = np.hstack([ fv_kaze, fv_histogram ]) # , fv_haralick, fv_hu_moments

    # scale features in the range (0-1)
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # reshaped_features = np.reshape(global_feature, (1, len(global_feature)))
    # rescaled_feature = scaler.fit_transform(reshaped_features)
    rescaled_feature = global_feature

    # predict label of test image
    prediction = clf.predict(rescaled_feature.reshape(1, -1))[0]
    print(prediction)
    # show predicted label on image
    cv2.putText(image, train_labels[prediction], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)

    # display the output image
    # fig, ax = plt.subplots(1, 1)
    plt.imsave(os.path.join(result_path, '%s.png' % idx), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # ax.savefig(os.path.join(result_path, '%s.jpg' % idx))
    # ax.show()
