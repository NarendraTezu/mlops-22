"""

    Author : Narendra Yadav
    Roll No : M20AIE263


"""



from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
# Standard scientific Python imports
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import sklearn
from skimage.transform import rescale, resize
from skimage import transform

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

#hyperparameter tuning
Gamma_list=[0.01 ,0.001, 0.0001, 0.0005]
c_list=[.1 ,.5, .4, 10, 5, 1]

h_param_comb=[{'gamma':g,'C':c} for g in Gamma_list for c in c_list]

digits = datasets.load_digits()


import numpy as np 

    

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
print(digits.images[-1].shape)
train_frac=0.8
test_frac=0.1
dev_frac=0.1
#print(data.shape)
# Split data into 80% train,10% validate and 10% test subsets
dev_test_frac=1-train_frac

X_train, X_dev_test, y_train, y_dev_test = train_test_split(data ,digits.target, test_size=dev_test_frac, shuffle=True,random_state=42)
X_test, X_dev, y_test, y_dev = train_test_split(X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True,random_state=42)

best_acc=-1
best_model=None
best_h_params=None 
for com_hyper in h_param_comb:

    # Create a classifier: a support vector classifier
    clf = svm.SVC()

    #Setting hyperparameter
    hyper_params=com_hyper
    clf.set_params(**hyper_params)
    #print(com_hyper)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted_train = clf.predict(X_train)
    predicted_dev = clf.predict(X_dev)
    predicted_test = clf.predict(X_test)
    
    #print("shape : ",predicted_dev.shape)
    
    cur_acc_train=metrics.accuracy_score(y_pred=predicted_train,y_true=y_train)
    cur_acc_dev=metrics.accuracy_score(y_pred=predicted_dev,y_true=y_dev)
    cur_acc_test=metrics.accuracy_score(y_pred=predicted_test,y_true=y_test)
    
    
    if cur_acc_dev>best_acc:
         best_acc=cur_acc_dev
         best_model=clf
         best_h_params=com_hyper
         print("found new best acc with: "+str(com_hyper))
         print("New best accuracy:"+ " train" + "  "+str(cur_acc_train)+ " "+ "dev" + " "+str(cur_acc_dev)+ " "+ "test" + " " +str(cur_acc_test))
         
predicted = best_model.predict(X_test)

print("****************************************************************************************************************************************")

print("Best hyperparameters were: ")
print(com_hyper)
print("Best accuracy on dev: ")
print(best_acc)

print("Question No 2 part 1")

print(digits.images[0].shape)

print("****************************************************************************************************************************************")

def resize_a(image,n):
    image = resize(image, (image.shape[0] // n, image.shape[1] // n),anti_aliasing=True)
    return image

digits_4 = np.zeros((1797, 2, 2))  # image divide by 4
digits_2 = np.zeros((1797, 4, 4))  # image divide by 2
digits_5 = np.zeros((1797, 4, 4))  # image divide by 5

for i in range(0,1797):
    digits_4[i] = resize_a(digits.images[i],4)

for i in range(0,1797):
    digits_2[i] = resize_a(digits.images[i],2)

for i in range(0,1797):
    digits_5[i] = resize_a(digits.images[i],5)


n_samples = len(digits.images)
data = digits_2.reshape((n_samples, -1))
#print(digits.images[-1].shape)
train_frac=0.8
test_frac=0.1
dev_frac=0.1
#print(data.shape)
# Split data into 80% train,10% validate and 10% test subsets
dev_test_frac=1-train_frac

X_train, X_dev_test, y_train, y_dev_test = train_test_split(data ,digits.target, test_size=dev_test_frac, shuffle=True,random_state=42)
X_test, X_dev, y_test, y_dev = train_test_split(X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True,random_state=42)

best_acc=-1
best_model=None
best_h_params=None 
for com_hyper in h_param_comb:

    # Create a classifier: a support vector classifier
    clf = svm.SVC()

    #Setting hyperparameter
    hyper_params=com_hyper
    clf.set_params(**hyper_params)
    #print(com_hyper)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted_train = clf.predict(X_train)
    predicted_dev = clf.predict(X_dev)
    predicted_test = clf.predict(X_test)
    
    #print("shape : ",predicted_dev.shape)
    
    cur_acc_train=metrics.accuracy_score(y_pred=predicted_train,y_true=y_train)
    cur_acc_dev=metrics.accuracy_score(y_pred=predicted_dev,y_true=y_dev)
    cur_acc_test=metrics.accuracy_score(y_pred=predicted_test,y_true=y_test)
    
    
    if cur_acc_dev>best_acc:
         best_acc=cur_acc_dev
         best_model=clf
         best_h_params=com_hyper
         print("found new best acc with: "+str(com_hyper))
         print("New best accuracy:"+ " train" + "  "+str(cur_acc_train)+ " "+ "dev" + " "+str(cur_acc_dev)+ " "+ "test" + " " +str(cur_acc_test))
         
predicted = best_model.predict(X_test)

print("Best hyperparameters were: ")
print(com_hyper)
print("Best accuracy on dev: ")
print(best_acc)

print("****************************************************************************************************************************************")
n_samples = len(digits.images)
data = digits_4.reshape((n_samples, -1))
#print(digits.images[-1].shape)
train_frac=0.8
test_frac=0.1
dev_frac=0.1
#print(data.shape)
# Split data into 80% train,10% validate and 10% test subsets
dev_test_frac=1-train_frac

X_train, X_dev_test, y_train, y_dev_test = train_test_split(data ,digits.target, test_size=dev_test_frac, shuffle=True,random_state=42)
X_test, X_dev, y_test, y_dev = train_test_split(X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True,random_state=42)

best_acc=-1
best_model=None
best_h_params=None 
for com_hyper in h_param_comb:

    # Create a classifier: a support vector classifier
    clf = svm.SVC()

    #Setting hyperparameter
    hyper_params=com_hyper
    clf.set_params(**hyper_params)
    #print(com_hyper)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted_train = clf.predict(X_train)
    predicted_dev = clf.predict(X_dev)
    predicted_test = clf.predict(X_test)
    
    #print("shape : ",predicted_dev.shape)
    
    cur_acc_train=metrics.accuracy_score(y_pred=predicted_train,y_true=y_train)
    cur_acc_dev=metrics.accuracy_score(y_pred=predicted_dev,y_true=y_dev)
    cur_acc_test=metrics.accuracy_score(y_pred=predicted_test,y_true=y_test)
    
    
    if cur_acc_dev>best_acc:
         best_acc=cur_acc_dev
         best_model=clf
         best_h_params=com_hyper
         print("found new best acc with: "+str(com_hyper))
         print("New best accuracy:"+ " train" + "  "+str(cur_acc_train)+ " "+ "dev" + " "+str(cur_acc_dev)+ " "+ "test" + " " +str(cur_acc_test))
         
predicted = best_model.predict(X_test)

print("Best hyperparameters were: ")
print(com_hyper)
print("Best accuracy on dev: ")
print(best_acc)


print("****************************************************************************************************************************************")

n_samples = len(digits.images)
data = digits_5.reshape((n_samples, -1))
#print(digits.images[-1].shape)
train_frac=0.8
test_frac=0.1
dev_frac=0.1
#print(data.shape)
# Split data into 80% train,10% validate and 10% test subsets
dev_test_frac=1-train_frac

X_train, X_dev_test, y_train, y_dev_test = train_test_split(data ,digits.target, test_size=dev_test_frac, shuffle=True,random_state=42)
X_test, X_dev, y_test, y_dev = train_test_split(X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True,random_state=42)

best_acc=-1
best_model=None
best_h_params=None 
for com_hyper in h_param_comb:

    # Create a classifier: a support vector classifier
    clf = svm.SVC()

    #Setting hyperparameter
    hyper_params=com_hyper
    clf.set_params(**hyper_params)
    #print(com_hyper)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted_train = clf.predict(X_train)
    predicted_dev = clf.predict(X_dev)
    predicted_test = clf.predict(X_test)
    
    #print("shape : ",predicted_dev.shape)
    
    cur_acc_train=metrics.accuracy_score(y_pred=predicted_train,y_true=y_train)
    cur_acc_dev=metrics.accuracy_score(y_pred=predicted_dev,y_true=y_dev)
    cur_acc_test=metrics.accuracy_score(y_pred=predicted_test,y_true=y_test)
    
    
    if cur_acc_dev>best_acc:
         best_acc=cur_acc_dev
         best_model=clf
         best_h_params=com_hyper
         print("found new best acc with: "+str(com_hyper))
         print("New best accuracy:"+ " train" + "  "+str(cur_acc_train)+ " "+ "dev" + " "+str(cur_acc_dev)+ " "+ "test" + " " +str(cur_acc_test))
         
predicted = best_model.predict(X_test)

print("Best hyperparameters were: ")
print(com_hyper)
print("Best accuracy on dev: ")
print(best_acc)
