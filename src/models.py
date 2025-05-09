from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def calculate_specificity(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn = cm[0, 0]  
    fp = cm[0, 1] 
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return specificity

def KnnClassifier(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    specificity = calculate_specificity(y_test, y_pred)
    
    return precision, accuracy, specificity, f1

def naive_bayes_classifier(X_train, y_train, X_test, y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    specificity = calculate_specificity(y_test, y_pred)
    
    return precision, accuracy, specificity, f1

def decision_tree_classifier(X_train, y_train, X_test, y_test):
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    specificity = calculate_specificity(y_test, y_pred)
    
    return precision, accuracy, specificity, f1

def logistic_regression_classifier(X_train, y_train, X_test, y_test):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    specificity = calculate_specificity(y_test, y_pred)
    
    return precision, accuracy, specificity, f1

def mlp_classifier(X_train, y_train, X_test, y_test):
    mlp = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    specificity = calculate_specificity(y_test, y_pred)
    
    return precision, accuracy, specificity, f1