#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import tree 
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import pprint

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# starting with all the features
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 
'other', 'long_term_incentive', 'restricted_stock', 'director_fees','to_messages',
'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

with open("final_project_dataset.pkl", "r") as data_file:
	data_dict = pickle.load(data_file)

#exploring the dataset
print "no. of data points: ", len(data_dict)

n=0
k=0
for key,value  in data_dict.iteritems():
  if data_dict[key]["poi"]==True:
    n+=1
  else:
    k+=1
print "no. of poi's ",n
print "no. of non poi's", k

# no. of NaNs
print "printing NaN information below"
for feature in features_list:
    cnt=0
    for key in data_dict.keys():
        if data_dict[key][feature] == 'NaN':
            cnt+=1

    print feature + " -> " + str(cnt)


#getting data in the right format & splitting the target, features.
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# using SelectKBest to determine most important features
from sklearn.feature_selection import SelectKBest, f_classif
kbest = SelectKBest(k=10)
selected_features = kbest.fit_transform(features,labels)
features_selected=[features_list[i+1] for i in kbest.get_support(indices=True)]
print 'Features selected by SelectKBest:'
print features_selected

feature_scores = ['%.2f' % elem for elem in kbest.scores_ ]

original_features_selected_tuple=[(features_list[i+1], feature_scores[i]) for i in kbest.get_support(indices=True)]

print original_features_selected_tuple

#features_list = features_selected
#features_list.insert(0,'poi')
#print "completed kbest and going back: ", features_list

#Features selected by SelectKBest:
#['deferral_payments', 'total_payments', 'loan_advances', 'deferred_income', 'exercised_stock_options', 
#'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 
#'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers

features = ["salary", "bonus"]

data_dict.pop("TOTAL",0)
data_dict.pop('The Travel Agency In the Park',0)
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


### Task 3: Create new feature(s)

for key in data_dict.keys():
    try:
        data_dict[key]['fraction_from_this_person_to_poi'] = float(data_dict[key]['from_this_person_to_poi']
                                                              )/data_dict[key]['from_messages']
    except:
        data_dict[key]['fraction_from_this_person_to_poi'] = 'NaN'
        
    try:
        data_dict[key]['fraction_from_poi_to_this_person'] = float(data_dict[key]['from_poi_to_this_person']
                                                              )/data_dict[key]['to_messages']
    except:
        data_dict[key]['fraction_from_poi_to_this_person'] = 'NaN'

features_list.append('fraction_from_this_person_to_poi')
features_list.append('fraction_from_poi_to_this_person')
features_list.remove('from_this_person_to_poi')
features_list.remove('from_poi_to_this_person')


print "feature list so far:"
pprint.pprint (features_list)



### Store to my_dataset for easy export below.

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

scaler = MinMaxScaler()
features = scaler.fit_transform(features)

### K-Means clustering
k_clf = KMeans(n_clusters=2, tol =0.001)

### split training and testing data

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


### Naive Bayes
nb_clf = GaussianNB()
nb_clf.fit(features_train, labels_train)
nb_pred = nb_clf.predict(features_test)
print "Naive Bayes Recall Score" + str(recall_score(labels_test, nb_pred))
print "Naive Bayes Precision Score" + str(precision_score(labels_test, nb_pred))
print "Naive Bayes Accuracy Score" + str(nb_clf.score(features_test, labels_test))

### Decision tree
d_clf = tree.DecisionTreeClassifier()
parameters = {'criterion': ['gini','entropy'],
              'min_samples_leaf':[1,5,10],
              'min_samples_split':[2, 10, 20],
                'max_depth':[10,15,20,25,30],
                'max_leaf_nodes':[5,10,30]}

d_clf = GridSearchCV(d_clf,parameters)
d_clf.fit(features_train,labels_train)
d_pred = d_clf.predict(features_test)
accuracy_d = d_clf.score(features_test,labels_test)
print "Decision tree accuracy: ", accuracy_d
print "Decision tree precision score: ", precision_score(labels_test,d_pred)
print "Decision tree recall score: ", recall_score(labels_test,d_pred)


### ADABOOST

from sklearn.ensemble import AdaBoostClassifier
ab_clf = AdaBoostClassifier(algorithm='SAMME')
parameters = {'n_estimators': [10, 20, 30, 40, 50],
'algorithm': ['SAMME', 'SAMME.R'],
'learning_rate': [.5,.8, 1, 1.2, 1.5]}
ab_clf = GridSearchCV(ab_clf, parameters)
ab_clf.fit(features_train, labels_train)
ab_pred= ab_clf.predict(features_test)
accuracy_ab = ab_clf.score(features_test, labels_test)
print 'ADABOOST:'
print accuracy_ab
print "ADABOOST Recall Score" + str(recall_score(labels_test, ab_pred))
print "ADABOOST Precision Score" + str(precision_score(labels_test, ab_pred))

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!



# creating instance of sss
sk_fold = StratifiedShuffleSplit(labels, 100, random_state = 42)

skb = SelectKBest (k=10)
pipe = Pipeline(steps=[('scaling',scaler),("SKB", skb), ("NaiveBayes", GaussianNB())])
parameters = {'SKB__k': range(1,10)}
# using this cross validation method in GridSearchCV
gs1 = GridSearchCV(pipe, param_grid = parameters, cv=sk_fold,scoring = 'f1')
gs1.fit(features, labels)
#extract best algorithm
clf = gs1.best_estimator_

#print 'best algorithm using strat_s_split'
print "best estimator from NB:", clf

skb_step = gs1.best_estimator_.named_steps['SKB']
print "best estimate for k", skb_step

pipe1 = Pipeline(steps=[('scaling',scaler),("SKB", skb),("pca", PCA(n_components = 0.95)),('tree', tree.DecisionTreeClassifier())])
parameters = {'SKB__k': range(4,10),
              'pca__n_components': range(1,5),
              'tree__random_state': [45],
              'tree__criterion': ('gini','entropy')}

gs2 = GridSearchCV(pipe1, param_grid = parameters, cv=sk_fold,scoring = 'f1')
gs2.fit(features, labels)
clf2 = gs2.best_estimator_
print "best estimator from DT:", clf2

pipe2 = Pipeline(steps=[('scaling',scaler),("SKB", skb),("pca", PCA(n_components = 0.95)),('adaboost',AdaBoostClassifier()) ])
parameters = {'adaboost__n_estimators': [5,10, 20, 30, 40, 50],
'adaboost__algorithm': ['SAMME', 'SAMME.R'],
'adaboost__learning_rate': [.5,.8, 1, 1.2, 1.5,2,2.2]}
gs3 = GridSearchCV(pipe2, param_grid = parameters, cv=sk_fold,scoring = 'f1')
gs3.fit(features,labels)
clf3 = gs3.best_estimator_

skb = clf2.named_steps['SKB']


# Get SelectKBest scores, rounded to 2 decimal places, name them "feature_scores"
feature_scores = ['%.2f' % elem for elem in skb.scores_ ]
# Get SelectKBest pvalues, rounded to 3 decimal places, name them "feature_scores_pvalues"
feature_scores_pvalues = ['%.3f' % elem for elem in  skb.pvalues_ ]
# Get SelectKBest feature names, whose indices are stored in 'skb.get_support',
# create a tuple of feature names, scores and pvalues, name it "features_selected_tuple"
features_selected_tuple=[(features_list[i+1], feature_scores[i], feature_scores_pvalues[i]) for i in skb.get_support(indices=True)]

# Sort the tuple by score, in reverse order
features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)

# Print
print ' '
print 'Selected Features, Scores, P-Values'
print features_selected_tuple

from tester import test_classifier

test_classifier(gs1.best_estimator_, my_dataset, features_list)
test_classifier(gs2.best_estimator_, my_dataset, features_list)
test_classifier(gs3.best_estimator_, my_dataset, features_list)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
