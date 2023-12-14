#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORTING LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


#IMPORTING THE DATA SET
data= pd.read_csv('survey_lung_cancer.csv')
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


# In[3]:


print(X)


# In[4]:


print(y)


# In[5]:


#DISPLAYING THE DATA TO CONFIRM WHETHER IT IS 'READ' PROPERLY
data.head(10)


# In[6]:


#VISUALIZING THE INDIVIDUAL ATTRIBUTES(I.E, INDEPENDENT VARIBALES)
data.hist(bins=50, figsize=(20,15))
plt.show()


# In[7]:


#Statistical Interpretation of data
data.describe()


# In[8]:


#Information regarding all the attributes in the dataframe
data.info()


# In[9]:


#Finding correlation between ["CHEST PAIN"] and all the other attributes in the data frame
corr_matrix= data.corr()
corr_matrix["CHEST PAIN"].sort_values(ascending=True)


# In[10]:


#Finding the correlation between ["ALCOHOL CONSUMING"] and all the other attributes in the data frame
corr_matrix["ALCOHOL CONSUMING"].sort_values(ascending=True)


# In[11]:


#Finding the correlation between ["YELLOW_FINGERS"] and all the other attributes in the data frame
corr_matrix["YELLOW_FINGERS"].sort_values(ascending=True)


# In[12]:


#Finding the correlation between "AGE" and all the other attributes in the dataframe
corr_matrix["AGE"].sort_values(ascending=True)


# In[13]:


#Finding Categorical Varibles 
s=(data.dtypes=='object')
object_cols= list(s[s].index)
print("Categorical variables that are present in the data frame")
print(object_cols)


# In[14]:


map1= {'MALE':0, 'FEMALE':1}
data['GENDER']= data['GENDER'].replace(map1)
data.head(5)


# In[15]:


data.head(5)


# In[16]:


data.shape


# In[17]:


data.duplicated().sum()


# In[18]:


data= data.drop_duplicates()


# In[19]:


data.duplicated().sum()


# In[20]:


from sklearn.preprocessing import LabelEncoder
LE= LabelEncoder()
data1= data.copy(deep=True)


# In[21]:


data1.head()


# In[22]:


data1.GENDER= LE.fit_transform(data1.GENDER)
data1.GENDER


# In[23]:


data1.LUNG_CANCER= LE.fit_transform(data1.LUNG_CANCER)
data1.LUNG_CANCER


# In[24]:


X= data1.iloc[:, :-1].values
y= data1.iloc[:, -1].values


# In[25]:


import seaborn as sns
sns.set_style("whitegrid")
sns.pairplot(data1, hue='LUNG_CANCER', height=2.5)
plt.show()


# In[26]:


#Splitting the data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=21)


# In[27]:


X_train.shape


# In[28]:


X_test.shape


# In[29]:


y_train.shape


# In[30]:


y_test.shape


# In[31]:


X_train


# In[32]:


y_train


# # LINEAR REGRESSION

# In[33]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
regressor= LinearRegression()
regressor.fit(X_train, y_train)


# In[34]:


y_pred= regressor.predict(X_test)


# In[35]:


print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# In[36]:


model= LinearRegression()
model.fit(X, y)


# In[37]:


predictions= model.predict(X)
print("shape of X:", X.shape)
print("shape of y:", y.shape)


# # LOGISTIC REGRESSION MODEL

# In[39]:


#LOGISTIC REGRESSION MODEL CREATION
from sklearn.linear_model import LogisticRegression
regressor_1= LogisticRegression()


# In[40]:


#TRAINING THE MODEL
regressor_1.fit(X_train, y_train)


# In[41]:


#PREDICTING THE VALUES 
y_pred_1= regressor_1.predict(X_test)


# In[42]:


print(np.concatenate((y_pred_1.reshape(len(y_pred_1),1),y_test.reshape(len(y_test),1)),1))


# In[43]:


#CALCULATING ACCURACY, CONFUSION_MATRIX FOR THE PREDICTIONS DONE FOR LOGISTIC REGRESSION
from sklearn.metrics import confusion_matrix, accuracy_score
cm1= confusion_matrix(y_test, y_pred_1)
print(cm1)


# In[44]:


accuracy_score(y_test, y_pred_1)


# In[45]:


sns.heatmap(cm1, annot=True, linewidth=0.7, linecolor='cyan', cmap='YlGnBu')
plt.title('Logistic Regression Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()


# In[46]:


#PRINTING THE MISS-CLASSIFIED SAMPLES
miss1= X_test[y_test != y_pred_1]


# In[47]:


miss1


# In[48]:


#DISPLAYING CLASSIFICATION REPORT FOR LOGISTIC REGRESSION MODEL
from sklearn.metrics import classification_report
cr1= classification_report(y_test, y_pred_1)


# In[49]:


print(cr1)


# In[50]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[146]:


from sklearn.metrics import roc_curve, auc
y_probs_1 = regressor_1.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs_1)

# Calculate Area Under the Curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression')
plt.legend(loc='lower right')
plt.show()


# # SUPPORT VECTOR MACHINE(SVM-CLASSIFIER)

# In[147]:


from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier 
from sklearn.svm import SVC


# In[148]:


#SUPPORT VECTOR MACHINE(SVM) MODEL CREATION
classifier_svm = OneVsRestClassifier(BaggingClassifier(SVC(C=10,kernel='rbf',random_state=9, probability=True),n_jobs=-1))


# In[149]:


#TRAINING THE SVM MODEL
classifier_svm.fit(X_train, y_train)


# In[150]:


#PREDICTING THE VALUES
y_pred_2= classifier_svm.predict(X_test)
y_probs_2= classifier_svm.predict_proba(X_test)[:,1]


# In[ ]:





# In[152]:


print(np.concatenate((y_pred_2.reshape(len(y_pred_2),1),y_test.reshape(len(y_test),1)),1))


# In[153]:


#CALCULATING THE ACCURACY SCORE AND CONFUSION SCORE FOR THE PREDICTED VALUES
cm2= confusion_matrix(y_test, y_pred_2)
print(cm2)


# In[154]:


accuracy_score(y_test, y_pred_2)


# In[155]:


sns.heatmap(cm2, annot=True, linewidth=0.7, linecolor='cyan', cmap='YlGnBu')
plt.title('svm Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()


# In[156]:


miss2= X_test[y_test != y_pred_2]


# In[157]:


miss2


# In[158]:


#DISPLAYING CLASSIFICATION_REPORT FOR SVM
print(classification_report(y_test, y_pred_2))


# # NAIVE BAYES CLASSIFIER

# In[159]:


#MODELLING THE NAIVE BAYES CLASSIFIER
from sklearn.naive_bayes import GaussianNB
NB_classifier= GaussianNB()


# In[160]:


#TRAINING THE MODEL
NB_classifier.fit(X_train, y_train)


# In[161]:


#PREDICTING THE TARGET VALUES
y_pred_3= NB_classifier.predict(X_test)


# In[162]:


print(np.concatenate((y_pred_3.reshape(len(y_pred_3),1),y_test.reshape(len(y_test),1)),1))


# In[163]:


#CALCULATING THE CONFUSION MATRIX
cm3= confusion_matrix(y_test, y_pred_3)
print(cm3)


# In[164]:


sns.heatmap(cm3, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', cmap='YlGnBu')
plt.title("Naive Bayes Classificatiuon Confusion Matrix")
plt.xlabel('y Predict')
plt.ylabel('y test')
plt.show()


# In[165]:


miss3= X_test[y_test != y_pred_3]


# In[166]:


miss3


# In[167]:


accuracy_score(y_test, y_pred_3)


# In[168]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred_3))


# In[169]:


model = GaussianNB()
model.fit(X_train, y_train)

# Predict probabilities for the positive class
y_probs_3 = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs_3)

# Calculate Area Under the Curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Naive Bayes Classifier')
plt.legend(loc='lower right')
plt.show()


# # DECISION TREE CLASSIFIER

# In[170]:


from sklearn.tree import DecisionTreeClassifier


# In[171]:


#DEFINING THE MODEL
DT_classifier= DecisionTreeClassifier()


# In[172]:


#TRAINING THE MODEL
DT_classifier.fit(X_train, y_train)


# In[173]:


#PREDICTING THE TARGET VALUES
y_pred_4= DT_classifier.predict(X_test)
y_probs_4= DT_classifier.predict_proba(X_test)[:, 1]


# In[174]:


print(np.concatenate((y_pred_4.reshape(len(y_pred_4),1),y_test.reshape(len(y_test),1)),1))


# In[175]:


#CALCULATING THE CONFUSION MATRIX AND ACCURACY SCORE
cm4= confusion_matrix(y_test, y_pred_4)
print(cm4)
accuracy_score(y_test, y_pred_4)


# In[176]:


sns.heatmap(cm4, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', cmap="YlGnBu")
plt.title('Decision Tree Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()


# In[177]:


miss4= X_test[y_test != y_pred_4]


# In[178]:


miss4


# In[179]:


#CLASSIFICATION REPORT
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_4))


# In[180]:


fpr, tpr, thresholds = roc_curve(y_test, y_probs_4)

# Calculate Area Under the Curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Decision Tree Classifier')
plt.legend(loc='lower right')
plt.show()


# # RANDOM FOREST CLASSIFIER

# In[181]:


#MODELLING THE RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
RM_classifier= RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
#TRAINING THE MODEL
RM_classifier.fit(X_train, y_train)


# In[182]:


#PREDICTING THE VALUES
y_pred_5= RM_classifier.predict(X_test)
y_probs_5= RM_classifier.predict_proba(X_test)[:, 1]


# In[183]:


print(np.concatenate((y_pred_5.reshape(len(y_pred_5),1),y_test.reshape(len(y_test),1)),1))


# In[184]:


#CALCULATING THE CONFUSION MATRIX AND ACCURACY SCORE
cm5= confusion_matrix(y_test, y_pred_5)
print(cm5)
accuracy_score(y_test, y_pred_5)


# In[185]:


sns.heatmap(cm5, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', cmap="YlGnBu")
plt.title('Random Forest Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()


# In[186]:


miss5= X_test[y_test!=y_pred_5]


# In[187]:


miss5


# In[188]:


print(classification_report(y_test, y_pred_5))


# In[189]:


# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs_5)

# Calculate Area Under the Curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest Classifier')
plt.legend(loc='lower right')
plt.show()


# # K NEAREST NEIGHBORS CLASSIFIER

# In[190]:


from sklearn.neighbors import KNeighborsClassifier
#MODELLING K-NEAREST NEIGHBORS CLASSIFIER
KNN_classifier= KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=2)
#TRAINING THE MODEL
KNN_classifier.fit(X_train, y_train)


# In[191]:


#PREDICTING THE VALUES
y_pred_6= KNN_classifier.predict(X_test)
y_probs_6= KNN_classifier.predict_proba(X_test)[:, 1]


# In[192]:


print(np.concatenate((y_pred_6.reshape(len(y_pred_6),1),y_test.reshape(len(y_test),1)),1))


# In[193]:


#CALCULATING THE CONFUSION MATRIX AND ACCURACY SCORE
cm6=confusion_matrix(y_test, y_pred_6)
print(cm6)
accuracy_score(y_test, y_pred_6)


# In[194]:


sns.heatmap(cm6, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', cmap="YlGnBu")
plt.title('KNN Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()


# In[195]:


miss6= X_test[y_test!=y_pred_6]


# In[196]:


miss6


# In[197]:


print(classification_report(y_test, y_pred_6))


# In[199]:


# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs_6)
k_neighbors= 7

# Calculate Area Under the Curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve for KNN (k={k_neighbors}) Classifier')
plt.legend(loc='lower right')
plt.show()


# # LASSO AND RIDGE REGRESSION

# In[215]:


from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled= scaler.transform(X_test)

lasso_alpha = 0.1
lasso = Lasso(alpha=lasso_alpha)
lasso.fit(X_train_scaled, y_train)
lasso_coef = lasso.coef_
print(lasso_coef)


# In[216]:


ridge_alpha = 1.0
ridge = Ridge(alpha=ridge_alpha)
ridge.fit(X_train_scaled, y_train)
ridge_coef = ridge.coef_
print(ridge_coef)


# In[217]:


plt.plot(lasso_coef, marker='o', label='Lasso Coefficients')
plt.plot(ridge_coef, marker='x', label='Ridge Coefficients')
plt.title("Lasso and Ridge Regression Coefficients")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.legend()
plt.show()


# # COMPARING ALL THE TEST SCORES

# In[200]:


#SCORE FOR LOGISTIC REGRESSION
score_logreg = regressor_1.score(X_test, y_test)
print(score_logreg)


# In[201]:


#SCORE FOR SVM CLASSIFIER
score_svmcla = classifier_svm.score(X_test, y_test)
print(score_svmcla)


# In[202]:


#SCORE FOR NAIVE BAYES CLASSIFIER
score_nbcla = NB_classifier.score(X_test, y_test)
print(score_nbcla)


# In[203]:


#SCORE FOR DECISION TREE CLASSIFIER
score_dtcla = DT_classifier.score(X_test, y_test)
print(score_dtcla)


# In[204]:


#SCORE FOR RANDOM FOREST CLASSIFIER
score_rfcla = RM_classifier.score(X_test, y_test)
print(score_rfcla)


# In[205]:


#SCORE FOR KNEAREST NEIGHBOR CLASSIFIER
score_knncla= KNN_classifier.score(X_test, y_test)
print(score_knncla)


# In[206]:


testscores=pd.DataFrame({'Logistic Regression Score':score_logreg,
                         'Support Vector Machine Score':score_svmcla,
                         'Naive Bayes Score':score_nbcla,
                         'Decision Tree Score':score_dtcla,
                         'Random Forest Score':score_rfcla,
                         'K-Nearest Neighbour Score':score_knncla},index=[0])

testscores


# # KMEANS WITH KNN
# 
# 

# In[220]:


pip install scikit-plot


# In[223]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools 
import csv
import math 
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import KMeans
from pylab import scatter,show,plot
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from pandas.plotting import parallel_coordinates
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score 
from itertools import cycle 
from sklearn.metrics import precision_score,recall_score,accuracy_score


# In[228]:


le = preprocessing.LabelEncoder()


# In[230]:


for col in data.columns.values: 
    if data[col].dtypes=='object': 
        le.fit(data[col]) 
        data[col]=le.transform(data[col])


# In[231]:


nclusters=3
kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(data)
class_labels = kmeans.labels_ 
print('class labels ',class_labels)
data2 = data
idx = 11 
data2.insert(loc=idx, column = 'Class', value = class_labels)


# In[232]:


data.head()


# In[233]:


neigh = KNeighborsClassifier(n_neighbors=int(math.sqrt(nclusters)))


# In[234]:


neigh.fit(X_train, y_train) 
y_pred_7 = neigh.predict(X_test)


# In[235]:


print('Total No. of Data',len(data))
print('70% of Training Data',len(X_train)) 
print('30% of Testing Data',len(X_test)) 
print('Predicted Output', neigh.predict(X_test)) 
print('Actual Output' , y_test)


# In[237]:


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize: 
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
        print("Normalized confusion matrix") 
    else: 
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap) 
    plt.title(title) 
    plt.colorbar()


# In[243]:


cnf_matrix = confusion_matrix(y_test, y_pred_7)
print(cnf_matrix)


# In[245]:


class_names = [0,1] 
# Plot non-normalized confusion matrix
plt.figure() 
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, # title='Normalized confusion matrix')
plt.show() 
target_names = ['class 0', 'class 1']
graph_cl = [0,1]
print('KMeans & KNN Classifier Performance Metrics')
print(classification_report(y_test, y_pred_7, target_names=target_names)) 
precision = precision_score(y_test, y_pred_7, average=None)
#print(precision)
recall = recall_score(y_test, y_pred_7, average=None) 
#rint(recall) 
accuracy = accuracy_score(y_test, y_pred_7) 
print('accuracy:',accuracy)


# In[250]:


kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
labels = kmeans.labels_
X_train, X_test, labels_train, labels_test = train_test_split(X, labels, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, labels_train)


# In[251]:


probas_7 = knn.predict_proba(X_test)

# Step 5: Compute ROC curve and area under the curve
fpr, tpr, thresholds = roc_curve(labels_test, probas_7[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for K-means with KNN')
plt.legend(loc='lower right')
plt.show()


# # FUZZY C-MEANS WITH KNN

# In[253]:


pip install scikit-fuzzy


# In[256]:


from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz 
import pandas as pd
import itertools
import csv 
from sklearn import preprocessing
from sklearn.cluster import KMeans
from pylab import scatter,show,plot
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from pandas.plotting import parallel_coordinates
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve 
from sklearn.metrics import average_precision_score
from itertools import cycle
from sklearn.metrics import precision_score,recall_score,accuracy_score
from sklearn.svm import SVC


# In[257]:


cntr, u_orig, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data, 3, 2, error=0.005, maxiter=1000, init=None)


# In[258]:


labels = [] 
for i in range(0,len(cntr[0])):
    a = max(cntr[0][i],cntr[1][i],cntr[2][i])
    if cntr[0][i] == a:
        labels.append(0)
    elif cntr[1][i] == a: 
        labels.append(1)
    elif cntr[2][i] == a: 
        labels.append(2)


# In[262]:


data2 = data
idx = 11
data2.insert(loc=idx,column='prediction', value = labels) # to insert new column in dataframe


# In[269]:


def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize: 
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title) 
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label') 
    plt.xlabel('Predicted label')


# In[270]:


plot_confusion_matrix


# In[288]:


clf = OneVsRestClassifier(BaggingClassifier(SVC(C=10,kernel='rbf',random_state=9, probability=True),n_jobs=-1))
X_train, X_test, y_train, y_test = train_test_split(data,labels, test_size=0.30, random_state=42) 
clf.fit(X_train, np.ravel(y_train))


# In[292]:


y_pred_9 = clf.predict(X_test)

print('Total No. of Data',len(data))
print('70% of Training Data',len(X_train))
print('30% of Testing Data',len(X_test)) 
print('Predicted Output', clf.predict(X_test))
print('Actual Output' , y_test)


# In[294]:


target_names = ['class 1', 'class 2'] 
graph_cl = [1,2] 
print('SVM Classifier Performance Metrics')

precision = precision_score(y_test, y_pred_9, average=None)
print('Precision: ',precision)
recall = recall_score(y_test, y_pred, average=None) 
print('Recall: ',recall)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ',accuracy)


# In[296]:


cnf_matrix = confusion_matrix(y_pred,y_test)
class_names = ['Class 1','Class 2']
# Plot non-normalized confusion matrix 
plt.figure() 
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization') 
plt.show()


# # KMEANS WITH SVM

# In[298]:


from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=2, random_state=42, cluster_std=2.0) 
kmeans = KMeans(n_clusters=2) 
X_clustered = kmeans.fit_predict(X)


# In[299]:


distances = kmeans.transform(X)
outlier_scores = np.min(distances, axis=1)
outlier_threshold = 5.0
potential_outliers = X[outlier_scores > outlier_threshold]
outlier_labels = np.ones(len(potential_outliers)) 
regular_labels = np.zeros(len(X) - len(potential_outliers)) 
y_extended = np.concatenate([regular_labels, outlier_labels])


# In[302]:


from sklearn.svm import OneClassSVM
X_train, X_test, y_train, y_test = train_test_split(X, y_extended, test_size=0.30, random_state=42) 
svm_outlier_detector = OneClassSVM(nu=0.1) 
svm_outlier_detector.fit(X_train) 
outlier_labels = np.ones(len(potential_outliers)) 
regular_labels = np.zeros(len(X) - len(potential_outliers))
y_extended = np.concatenate([regular_labels, outlier_labels])


# In[303]:


X_train, X_test, y_train, y_test = train_test_split(X, y_extended, test_size=0.30, random_state=42)
svm_outlier_detector = OneClassSVM(nu=0.1) 
svm_outlier_detector.fit(X_train)


# In[305]:


y_pred_outliers = svm_outlier_detector.predict(potential_outliers)
y_pred_regular = svm_outlier_detector.predict(X_test)
y_pred_combined = np.concatenate([y_pred_regular, y_pred_outliers])
y_pred_combined_binary = np.where(y_pred_combined == 1, 1, 0)[:len(y_test)] 
# Ensure lengths match 
accuracy = accuracy_score(y_test, y_pred_combined_binary)
conf_matrix = confusion_matrix(y_test, y_pred_combined_binary) 
print("Confusion Matrix:")
print(conf_matrix) 
fpr, tpr, _ = roc_curve(y_test, y_pred_combined_binary)
roc_auc = auc(fpr, tpr)


# In[306]:


print(f"Accuracy: {accuracy}") 
print(f"ROC AUC: {roc_auc}")


# In[307]:


plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=X_clustered, cmap='viridis', label='Data Points')
plt.scatter(potential_outliers[:, 0], potential_outliers[:, 1], marker='o', s=100, color='black', label='Potential Outliers')
plt.title('Data Points and Potential Outliers') 
plt.xlabel('Feature 1') 
plt.ylabel('Feature 2') 
plt.legend() 
plt.subplot(1, 2, 2)
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label') 
plt.ylabel('True Label') 
plt.tight_layout()
plt.show()


# In[ ]:




