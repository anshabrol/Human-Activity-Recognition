# Human Activity Recognition: Dimensionality Reduction Using Clustering

## Overview

This project demonstrates how dimensionality reduction techniques, specifically clustering, can be applied to Human Activity Recognition (HAR) data. The goal is to reduce the complexity of the dataset while preserving essential patterns that help in recognizing human activities. The code uses a clustering algorithm (K-Means) to reduce the number of features and improve the performance of classification models for human activity recognition.

## Project Structure

The main notebook in this repository is **`dimensionality_reduction_using_clustering.ipynb`**, which covers the following steps:

1. **Data Loading**: The HAR dataset is loaded, which contains features like accelerometer and gyroscope readings for various human activities (e.g., walking, sitting, etc.).
2. **Clustering**: K-Means clustering is applied to the dataset to identify patterns and group similar data points. This technique is used for dimensionality reduction by grouping data into clusters.
3. **Feature Selection**: The reduced features from clustering are selected to improve the performance of machine learning models.
4. **Model Training**: A classification model is trained using the reduced features to recognize different human activities.
5. **Evaluation**: The model's performance is evaluated using standard metrics like accuracy, precision, recall, and F1-score.

## Key Features

- **K-Means Clustering**: A widely used unsupervised machine learning technique to find similarities in data, which helps in reducing the dimensionality of the feature space.
- **Human Activity Recognition**: Classification of human activities based on sensor data such as accelerometer and gyroscope readings.
- **Model Evaluation**: Evaluation of the model's performance after dimensionality reduction and classification.

## Requirements

To run the notebook, you need the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `sklearn`

You can install these libraries using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/anshabrol/Human-Activity-Recognition-Dimensionality-Reduction-Using-Clustering.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Human-Activity-Recognition-Dimensionality-Reduction-Using-Clustering
    ```
3. Open the Jupyter notebook:
    ```bash
    jupyter notebook dimensionality_reduction_using_clustering.ipynb
    ```

4. Follow the steps in the notebook to load the data, apply clustering, and evaluate the classification model.

## Contributing

Feel free to fork the repository, create pull requests, or open issues for improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

