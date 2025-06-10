import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score 

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