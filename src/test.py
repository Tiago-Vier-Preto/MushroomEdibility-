import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preProcessingData import pre_process_data # Importando sua função
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay, RocCurveDisplay

# --- Configuração ---
# Tenta carregar o arquivo. Se não encontrar, cria um DataFrame de exemplo.
try:
    df_raw = pd.read_csv('data.csv', sep=';')
    print("Arquivo 'data.csv' carregado com sucesso.")
except FileNotFoundError:
    print("Aviso: O arquivo 'data.csv' não foi encontrado.")
    print("Gerando um conjunto de dados de exemplo para demonstração.")
    from sklearn.datasets import make_classification
    # Gera um dataset com características semelhantes para a demonstração
    X_sample, y_sample = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=777
    )
    df_raw = pd.DataFrame(X_sample)
    # Adiciona uma coluna de alvo no início para corresponder ao cenário de erro
    df_raw.insert(0, 'class', ['p' if i == 1 else 'e' for i in y_sample])


# --- 1. Preparação dos Dados ---

# Aplica a função de pré-processamento ao DataFrame completo PRIMEIRO.
# Isso garante que a remoção de linhas seja consistente para features e alvo.
df_processed = pre_process_data(df_raw)

# Agora, separa as features (X) e o alvo (y) do DataFrame JÁ processado.
# Assumindo que a primeira coluna é o alvo.
target_column = df_processed.columns[0]
y_text = df_processed[target_column]
X = df_processed.drop(columns=[target_column])

# Converte a coluna alvo de texto para numérico
# Mapeia 'p' (poisonous) para 1 (classe positiva) e 'e' (edible) para 0.
y = y_text.map({'p': 0, 'e': 1})
print("Coluna alvo convertida: 'p' -> 0, 'e' -> 1.")


# Divide os dados em conjuntos de treino e teste (80% treino, 20% teste)
# Agora X e y têm o mesmo número de amostras.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=777, stratify=y
)

# Padroniza as features.
# Assumindo que pre_process_data já tornou todas as colunas de X numéricas.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- 2. Definição dos Modelos com os Hiperparâmetros Especificados ---

models = {
    'KNN': KNeighborsClassifier(
        n_neighbors=3,
        weights='distance',
        metric='minkowski',
        p=5
    ),
    'Rede Neural': MLPClassifier(
        hidden_layer_sizes=(129, 18),
        activation='tanh',
        solver='adam',
        learning_rate_init=0.0035,
        alpha=1e-5,  # Pequeno valor para regularização L2
        random_state=777,
        max_iter=500 # Aumentar iterações para garantir convergência
    ),
    'Árvore de Decisão': DecisionTreeClassifier(
        max_depth=24,
        criterion='entropy',
        min_samples_split=13,
        min_samples_leaf=1,
        ccp_alpha=0.001, # Pequeno fator de poda
        random_state=777
    )
}

# --- 3. Treinamento, Predição e Geração dos Gráficos ---

# Cria uma figura para os plots com 3 linhas e 3 colunas
fig, axes = plt.subplots(3, 3, figsize=(20, 17))
fig.suptitle('Avaliação de Modelos: Matriz de Confusão, Curva PR e Curva ROC', fontsize=20)

for i, (model_name, model) in enumerate(models.items()):
    print(f"Treinando e avaliando o modelo: {model_name}...")

    # Usa dados escalados para KNN e Rede Neural
    if model_name in ['KNN', 'Rede Neural']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] # Probabilidade da classe positiva
    # Usa dados originais para a Árvore de Decisão
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    # --- Gráfico da Matriz de Confusão (Linha 1) ---
    ax_cm = axes[0, i]
    # Define a ordem dos rótulos para [1, 0] para inverter o eixo Y.
    labels = [1, 0] 
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp_cm.plot(ax=ax_cm, cmap='Blues', colorbar=False)
    ax_cm.set_title(f'Matriz de Confusão - {model_name}', fontsize=14)
    ax_cm.grid(False)

    # --- Gráfico da Curva Precision-Recall (Linha 2) ---
    ax_pr = axes[1, i]
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax_pr, name=model_name)
    ax_pr.set_title(f'Curva Precision-Recall - {model_name}', fontsize=14)
    ax_pr.grid(True)
    ax_pr.legend(loc='lower left')
    
    # --- Gráfico da Curva ROC (Linha 3) ---
    ax_roc = axes[2, i]
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax_roc, name=model_name)
    ax_roc.plot([0, 1], [0, 1], "k--", label="Chance") # Adiciona linha de referência
    ax_roc.set_title(f'Curva ROC - {model_name}', fontsize=14)
    ax_roc.grid(True)
    ax_roc.legend()


# Ajusta o layout para evitar sobreposição e exibe os gráficos
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("\nAnálise concluída.")
