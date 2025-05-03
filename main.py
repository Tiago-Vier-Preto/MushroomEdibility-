from src.utils import load_data
from src.preProcessingData import pre_process_data
from src.exploratoryDataAnalysis import full_feature_analysis
from src.train import train_model

def main():
    data = load_data('data.csv')

    data = pre_process_data(data)

    avg_metrics = train_model(data.drop(columns=['class']), data['class'])

    print("Average Metrics for each model:")
    for model, metrics in avg_metrics.items():
        print(f"{model}: {metrics}")

if __name__ == "__main__":
    main()