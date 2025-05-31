from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

RANDOM_STATE = 42

def calculate_specificity(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return specificity

def KnnClassifier(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(
        n_neighbors=3,
        weights='distance',
        metric='minkowski',
        p=5
    )
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    specificity = calculate_specificity(y_test, y_pred)
    return precision, accuracy, specificity, f1

def decision_tree_classifier(X_train, y_train, X_test, y_test):
    dtc = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=24,
        min_samples_split=13,
        min_samples_leaf=1,
        ccp_alpha=3.659264287811803e-05,
        random_state=RANDOM_STATE
    )
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    specificity = calculate_specificity(y_test, y_pred)
    return precision, accuracy, specificity, f1

def mlp_classifier(X_train, y_train, X_test, y_test):
    mlp = MLPClassifier(
        hidden_layer_sizes=(129, 18),
        activation='tanh',
        solver='adam',
        alpha=2.094694731278907e-05,
        learning_rate_init=0.0034857895430187376,
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=RANDOM_STATE
    )
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    specificity = calculate_specificity(y_test, y_pred)
    return precision, accuracy, specificity, f1