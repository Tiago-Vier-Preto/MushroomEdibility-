import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score 
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

def plot_feature_importance(importance_df):
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance_mean', y='feature', data=importance_df.sort_values(by="importance_mean", ascending=False))
    plt.title('Importancia de Atributos por Permutacao do KNN')
    plt.xlabel('Reducao Media na Precisao')
    plt.ylabel('Atributo')
    plt.tight_layout()
    plt.show()

def analyze_feature_importance_with_progress(knn_model, X_test, y_test, feature_names, n_repeats=3):
    print("\nIniciando Analise de Importancia de Atributos")
    y_pred_base = knn_model.predict(X_test)

    baseline_score = precision_score(y_test, y_pred_base, average='weighted', zero_division=0)

    importances = {}

    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_names)

    for i, feature in enumerate(feature_names):
        print(f"Analisando atributo {i+1}/{len(feature_names)}: {feature}")

        permuted_scores = []
        for r in range(n_repeats):

            X_test_permuted = X_test.copy()

            original_column = X_test_permuted[feature].values
            permuted_column = np.random.permutation(original_column)
            X_test_permuted[feature] = permuted_column

            y_pred_permuted = knn_model.predict(X_test_permuted)

            score = precision_score(y_test, y_pred_permuted, average='weighted', zero_division=0)
            permuted_scores.append(score)

        importance = baseline_score - np.mean(permuted_scores)
        importances[feature] = importance

    print("\nAnalise de Importancia Concluida")

    importance_df = pd.DataFrame.from_dict(importances, orient='index', columns=['importance_mean'])
    importance_df = importance_df.reset_index().rename(columns={'index': 'feature'})

    sorted_df = importance_df.sort_values(by="importance_mean", ascending=False)
    print("Importancia dos atributos:")

    lines = sorted_df.to_string().split('\n')
    print(lines[0]) 
    for row in lines[1:]:
        print(row)
        print()

    plot_feature_importance(importance_df)

    return importance_df

def plot_decision_boundary_2d(knn_model, X_train, y_train, features_2d, X_test=None, y_test=None):
    if len(features_2d) != 2:
        raise ValueError("Features_2d deve conter exatamente 2 nomes de atributos.")

    feature1, feature2 = features_2d[0], features_2d[1]
    print(f"\nGerando fronteira de decisao para: '{feature1}' vs '{feature2}'")

    X_train_2d = X_train[[feature1, feature2]]

    knn_2d = KNeighborsClassifier(
        n_neighbors=knn_model.n_neighbors,
        weights=knn_model.weights,
        metric=knn_model.metric,
        p=knn_model.p
    )
    knn_2d.fit(X_train_2d, y_train)

    h = .05  
    x_min, x_max = X_train_2d.iloc[:, 0].min() - 0.5, X_train_2d.iloc[:, 0].max() + 0.5
    y_min, y_max = X_train_2d.iloc[:, 1].min() - 0.5, X_train_2d.iloc[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))

    num_classes = len(np.unique(y_train))
    cmap_light = ListedColormap(['#AAFFAA', '#FFAAAA', '#AAAAFF'][:num_classes]) 
    colors_bold = ['green', 'red', 'blue'][:num_classes]

    plt.contourf(xx, yy, Z, cmap=cmap_light)

    if X_test is not None and y_test is not None:
        X_test_2d = X_test[[feature1, feature2]]
        sns.scatterplot(x=X_test_2d.iloc[:, 0], y=X_test_2d.iloc[:, 1], hue=y_test,
                        palette=colors_bold, alpha=0.8, edgecolor="k")

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"Fronteira de Decisão do KNN: {feature1} vs {feature2}")
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend(title='Classe')
    plt.show()