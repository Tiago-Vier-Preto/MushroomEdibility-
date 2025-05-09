import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Dados originais
data = {
    "Modelo": ["KNN"] * 3 + ["Naive Bayes"] * 3 + ["Árvore de Decisão"] * 3 + ["Regressão Logística"] * 3 + ["Rede Neural"] * 3,
    "Seed": [123, 65, 777] * 5,
    "Precisão": [
        0.99988514, 0.99988512, 0.99988513,
        0.71379917, 0.71517563, 0.71280006,
        0.99745595, 0.99757125, 0.99771874,
        0.77369661, 0.77382084, 0.77406906,
        0.99844293, 0.99813285, 0.99919615
    ],
    "Acurácia": [
        0.99988510, 0.99988510, 0.99988510,
        0.67588587, 0.67705142, 0.67495025,
        0.99745580, 0.99757071, 0.99771844,
        0.77343525, 0.77355033, 0.77381294,
        0.99844062, 0.99812877, 0.99919570
    ],
    "F1-Score": [
        0.99981639, 0.99977877, 0.99977968,
        0.83883812, 0.84045586, 0.83786511,
        0.99709282, 0.99720448, 0.99731357,
        0.75113095, 0.75159707, 0.75145518,
        0.99768195, 0.99712735, 0.99871057
    ],
    "Especificidade": [
        0.99988510, 0.99988510, 0.99988510,
        0.67162004, 0.67276842, 0.67065424,
        0.99745580, 0.99757069, 0.99771842,
        0.77354166, 0.77366102, 0.77391458,
        0.99844054, 0.99812866, 0.99919566
    ]
}

# Converte todos os valores para a faixa de 0 a 100
for metric in ["Precisão", "Acurácia", "F1-Score", "Especificidade"]:
    data[metric] = [value * 100 for value in data[metric]]

# Criar o DataFrame
df = pd.DataFrame(data)

# Função para exibir box e violin plot com porcentagem
def plot_box_and_violin(df, metric):
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    # Boxplot
    ax0 = plt.subplot(gs[0])
    sns.boxplot(x="Modelo", y=metric, data=df, ax=ax0)
    ax0.set_title(f'Box Plot - {metric}')
    ax0.set_ylabel(f"{metric} (%)")
    ax0.set_xlabel("Modelo")
    ax0.tick_params(axis='x', rotation=45)
    ax0.set_ylim(0, 100)

    # Violin plot
    ax1 = plt.subplot(gs[1])
    sns.violinplot(x="Modelo", y=metric, data=df, ax=ax1, inner="point")
    ax1.set_title(f'Violin Plot - {metric}')
    ax1.set_ylabel(f"{metric} (%)")
    ax1.set_xlabel("Modelo")
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(0, 100)

    plt.tight_layout()
    plt.show()

# Gerar plots para cada métrica
metrics = ["Precisão", "Acurácia", "F1-Score", "Especificidade"]
for metric in metrics:
    plot_box_and_violin(df, metric)
