import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preProcessingData import pre_process_data # Importando sua função
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay, accuracy_score, precision_score, f1_score

# --- Funções de Normalização e Modelos Fornecidas ---

def normalize_specific_columns(train_data, test_data, columns):
    """Normaliza colunas específicas de um conjunto de dados de treino e teste."""
    train_data = train_data.copy()
    test_data = test_data.copy()

    for column in columns:
        if column in train_data.columns:
            min_val = train_data[column].min()
            max_val = train_data[column].max()
            if max_val != min_val:
                train_data[column] = (train_data[column] - min_val) / (max_val - min_val)
                test_data[column] = (test_data[column] - min_val) / (max_val - min_val)
            else:
                train_data[column] = 0.0
                test_data[column] = 0.0
    return train_data, test_data

def calculate_specificity(y_true, y_pred):
    """Calcula a especificidade a partir da matriz de confusão."""
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0, 0] if cm.shape == (2, 2) else 0
    fp = cm[0, 1] if cm.shape == (2, 2) else 0
    return tn / (tn + fp) if (tn + fp) > 0 else 0

# --- Configuração ---
RANDOM_STATE = 42

# Tenta carregar o arquivo.
try:
    df_raw = pd.read_csv('data.csv', sep=';')
    print("Arquivo 'data.csv' carregado com sucesso.")
except FileNotFoundError:
    print("Aviso: O arquivo 'data.csv' não foi encontrado. Saindo.")
    exit()

# --- 1. Preparação dos Dados ---

df_processed = pre_process_data(df_raw)
target_column = df_processed.columns[0]
y = df_processed[target_column].map({'p': 1, 'e': 0})
X = df_processed.drop(columns=[target_column])
print("Coluna alvo convertida: 'p' -> 1, 'e' -> 0.")

# --- 2. Validação Cruzada e Treinamento ---

models_config = {
    'KNN': KNeighborsClassifier(n_neighbors=3, weights='distance', metric='minkowski', p=5),
    'Árvore de Decisão': DecisionTreeClassifier(criterion='entropy', max_depth=24, min_samples_split=13, min_samples_leaf=1, ccp_alpha=3.659264287811803e-05, random_state=RANDOM_STATE),
    'Rede Neural': MLPClassifier(hidden_layer_sizes=(129, 18), activation='tanh', solver='adam', alpha=2.094694731278907e-05, learning_rate_init=0.0034857895430187376, max_iter=1000, early_stopping=True, n_iter_no_change=10, random_state=RANDOM_STATE)
}

metrics = {model_name: [] for model_name in models_config.keys()}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)

print("\nIniciando validação cruzada (5-folds)...")

# Loop de validação cruzada
for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"--- Fold {fold+1}/5 ---")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    X_train_norm, X_test_norm = normalize_specific_columns(
        X_train, X_test, ['stem-width', 'stem-height', 'cap-diameter']
    )

    for model_name, model in models_config.items():
        # Usa dados normalizados para KNN e MLP, e originais para Árvore de Decisão
        current_X_train = X_train_norm if model_name != 'Árvore de Decisão' else X_train
        current_X_test = X_test_norm if model_name != 'Árvore de Decisão' else X_test

        model.fit(current_X_train, y_train)
        y_pred = model.predict(current_X_test)
        
        # Calcula métricas para este fold
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        spec = calculate_specificity(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        metrics[model_name].append([prec, acc, spec, f1])

# --- 3. Resultados e Geração dos Gráficos (com o último fold) ---

print("\n--- Resultados Médios da Validação Cruzada ---")
for model_name, results in metrics.items():
    avg_metrics = np.mean(results, axis=0)
    print(f"\n{model_name}:")
    print(f"  - Precisão Média:    {avg_metrics[0]:.4f}")
    print(f"  - Acurácia Média:    {avg_metrics[1]:.4f}")
    print(f"  - Especificidade Média: {avg_metrics[2]:.4f}")
    print(f"  - F1-Score Médio:      {avg_metrics[3]:.4f}")

# Prepara para gerar gráficos usando o último fold
print("\nGerando gráficos de avaliação usando o último fold como amostra...")
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Avaliação de Modelos (Amostra do Último Fold)', fontsize=20)

for i, (model_name, model) in enumerate(models_config.items()):
    # Re-treina e prediz no último fold para garantir que temos as probabilidades
    current_X_train = X_train_norm if model_name != 'Árvore de Decisão' else X_train
    current_X_test = X_test_norm if model_name != 'Árvore de Decisão' else X_test
    
    model.fit(current_X_train, y_train)
    y_pred_last_fold = model.predict(current_X_test)
    y_prob_last_fold = model.predict_proba(current_X_test)[:, 1]

    # Gráfico da Matriz de Confusão
    ax_cm = axes[0, i]
    labels = [1, 0]
    cm = confusion_matrix(y_test, y_pred_last_fold, labels=labels)
    disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp_cm.plot(ax=ax_cm, cmap='Blues', colorbar=False)
    ax_cm.set_title(f'Matriz de Confusão - {model_name}', fontsize=14)
    ax_cm.grid(False)

    # Gráfico da Curva Precision-Recall
    ax_pr = axes[1, i]
    PrecisionRecallDisplay.from_predictions(y_test, y_prob_last_fold, ax=ax_pr, name=model_name)
    ax_pr.set_title(f'Curva Precision-Recall - {model_name}', fontsize=14)
    ax_pr.grid(True)
    ax_pr.legend(loc='lower left')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("\nAnálise concluída.")
