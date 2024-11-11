import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def ana(csv_dir, png_dir):
    data = pd.read_csv(csv_dir)

    # # Assuming the target variable is named 'target'
    # data_aggregated = data.groupby(["Actual", "Predicted"], as_index=False).sum()

    # Pivot the DataFrame to create the confusion matrix
    confusion_matrix = data.pivot(index="Actual", columns="Predicted", values="nPredictions")

    # # Check for non-finite values (NaN or inf)
    # print("Non-finite values in confusion matrix:")
    # print(confusion_matrix[~np.isfinite(confusion_matrix)])

    # # Handle non-finite values by filling with 0 (or another appropriate value)
    # confusion_matrix = confusion_matrix.fillna(0)
    # confusion_matrix = confusion_matrix.astype(int)

    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')

    # 保存图像
    output_path = png_dir
    plt.savefig(output_path)
    plt.close()

if __name__ == '__main__':
    ana("conf_matrix_best_hype.csv", "conf_matrix_best_hype.png")