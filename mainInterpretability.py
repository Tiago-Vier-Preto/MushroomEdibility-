from src.utils import load_data
from src.preProcessingData import pre_process_data
from src.trainOpt import normalize_specific_columns

from src.interpretability import plot_decision_boundary_2d, analyze_feature_importance_with_progress
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def main():
    data = load_data('data.csv')
    data = pre_process_data(data)
    X = data.drop(columns=['class'])
    y = data['class']

    class_mapping = {'e': 0, 'p': 1} 
    y = y.map(class_mapping)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    feature_names = X_train.columns.tolist() 

    numeric_columns = ['stem-width', 'stem-height', 'cap-diameter']
    X_train_norm, X_test_norm = normalize_specific_columns(X_train.copy(), X_test.copy(), numeric_columns)

    print("Treinamento do KNN")
    final_knn = KNeighborsClassifier(
        n_neighbors=3,
        weights='distance',
        metric='minkowski',
        p=5
    )
    final_knn.fit(X_train_norm, y_train)

    print("Interpretabilidade do KNN")
    analyze_feature_importance_with_progress(
        knn_model=final_knn, 
        X_test=X_test_norm, 
        y_test=y_test, 
        feature_names=feature_names, 
        n_repeats=3
    )

    print("\nIniciando a geração das fronteiras de decisão")

    feature_pairs = list(combinations(numeric_columns, 2))

    for pair in feature_pairs:
        plot_decision_boundary_2d(
            knn_model=final_knn,
            X_train=X_train_norm,
            y_train=y_train,
            features_2d=list(pair),
            X_test=X_test_norm,
            y_test=y_test
        )

if __name__ == "__main__":
    main()