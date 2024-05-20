# Customer Purchase Prediction using DecisionTreeClassifier Task 03

This project aims to predict whether a customer will purchase a product or service based on their demographic and behavioral data. We use the Bank Marketing dataset from the UCI Machine Learning Repository for this purpose. The dataset contains various features such as age, job, marital status, education, and past marketing campaign interactions.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Libraries Used](#libraries-used)
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
7. [Data Preprocessing](#data-preprocessing)
8. [Model Training](#model-training)
9. [Model Evaluation](#model-evaluation)
10. [Conclusion](#conclusion)
11. [References](#references)

## Project Overview
This project implements a Decision Tree Classifier to predict whether a customer will purchase a product or service. The main steps involved are:
- Data Loading
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Model Training
- Model Evaluation

## Dataset
The dataset used is the Bank Marketing dataset from the UCI Machine Learning Repository. It includes information about customer demographics and interactions with past marketing campaigns.

- **Source**: [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- **Features**: 16 input features including age, job, marital status, education, etc.
- **Target**: Whether the client subscribed a term deposit (binary: 'yes','no')

## Installation
To run the project, ensure you have Python installed along with the necessary libraries. You can install the required libraries using the following command:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Usage
1. Clone the repository:
    ```bash
    git clone https://github.com/AnshPandey74/Prodigy_DS_03.git
    ```
2. Navigate to the project directory:
    ```bash
    cd customer-purchase-prediction
    ```
3. Open the Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
4. Run the `Prodigy_DS_03.ipynb` notebook.

## Libraries Used
- **Pandas**: Data manipulation and analysis.
- **Numpy**: Numerical computing.
- **Scikit-learn**: Machine learning library.
- **Matplotlib**: Plotting and visualization.
- **Seaborn**: Statistical data visualization.
- **Jupyter**: Interactive computing environment.

## Exploratory Data Analysis (EDA)
The EDA section provides insights into the dataset using various visualizations and statistical measures. Key steps include:
- Visualizing the distribution of features.
- Checking for missing values.
- Analyzing correlations between features and the target variable.

## Data Preprocessing
Data preprocessing involves:
- Handling missing values.
- Encoding categorical variables.
- Normalizing numerical features.
- Splitting the data into training and testing sets.

## Model Training
We use a `DecisionTreeClassifier` from the `scikit-learn` library. The model is trained on the preprocessed data, and hyperparameters are tuned for optimal performance.

### Steps:
1. Import the necessary libraries:
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    ```
2. Load and preprocess the dataset.
3. Split the data into training and testing sets.
    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    ```
4. Train the model:
    ```python
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    ```
5. Make predictions and evaluate the model:
    ```python
    y_pred = clf.predict(X_test)
    ```

## Model Evaluation
Model evaluation involves assessing the performance of the trained model using metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

### Example:
```python
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

## Conclusion
The Decision Tree Classifier provides a way to predict customer purchases based on demographic and behavioral data. Through this project, we demonstrate the process of building, training, and evaluating a machine learning model.

### Screenshots
![Correlation matrix heatmap](https://github.com/AnshPandey74/Prodigy_DS_03/raw/c8c4a73f890c3735a79817b8b8c6fb5b080943c0/screenshots/1.png)
![Survival count](https://github.com/AnshPandey74/Prodigy_DS_03/raw/c8c4a73f890c3735a79817b8b8c6fb5b080943c0/screenshots/2.png)
![Passenger class distribution](https://github.com/AnshPandey74/Prodigy_DS_03/raw/c8c4a73f890c3735a79817b8b8c6fb5b080943c0/screenshots/3.png)
![Gender distribution](https://github.com/AnshPandey74/Prodigy_DS_03/raw/c8c4a73f890c3735a79817b8b8c6fb5b080943c0/screenshots/4.png)
![Embarkation point distribution](https://github.com/AnshPandey74/Prodigy_DS_01/raw/c8c4a73f890c3735a79817b8b8c6fb5b080943c0/screenshots/5.png)
![Another visualization](https://github.com/AnshPandey74/Prodigy_DS_03/raw/c8c4a73f890c3735a79817b8b8c6fb5b080943c0/screenshots/6.png)
![Additional visualization](https://github.com/AnshPandey74/Prodigy_DS_03/raw/c8c4a73f890c3735a79817b8b8c6fb5b080943c0/screenshots/7.png)
![Further visualization](https://github.com/AnshPandey74/Prodigy_DS_03/raw/c8c4a73f890c3735a79817b8b8c6fb5b080943c0/screenshots/8.png)
![Further visualization](https://github.com/AnshPandey74/Prodigy_DS_03/raw/c8c4a73f890c3735a79817b8b8c6fb5b080943c0/screenshots/9.png)
![Further visualization](https://github.com/AnshPandey74/Prodigy_DS_03/raw/c8c4a73f890c3735a79817b8b8c6fb5b080943c0/screenshots/10.png)
![Further visualization](https://github.com/AnshPandey74/Prodigy_DS_03/raw/c8c4a73f890c3735a79817b8b8c6fb5b080943c0/screenshots/11.png)
![Further visualization](https://github.com/AnshPandey74/Prodigy_DS_03/raw/c8c4a73f890c3735a79817b8b8c6fb5b080943c0/screenshots/12.png)
![Further visualization](https://github.com/AnshPandey74/Prodigy_DS_03/raw/c8c4a73f890c3735a79817b8b8c6fb5b080943c0/screenshots/13.png)
![Further visualization](https://github.com/AnshPandey74/Prodigy_DS_03/raw/c8c4a73f890c3735a79817b8b8c6fb5b080943c0/screenshots/14.png)
![Further visualization](https://github.com/AnshPandey74/Prodigy_DS_03/raw/c8c4a73f890c3735a79817b8b8c6fb5b080943c0/screenshots/15.png)
![Further visualization](https://github.com/AnshPandey74/Prodigy_DS_03/raw/c8c4a73f890c3735a79817b8b8c6fb5b080943c0/screenshots/16.png)
![Further visualization](https://github.com/AnshPandey74/Prodigy_DS_03/raw/c8c4a73f890c3735a79817b8b8c6fb5b080943c0/screenshots/17.png)
![Further visualization](https://github.com/AnshPandey74/Prodigy_DS_03/raw/c8c4a73f890c3735a79817b8b8c6fb5b080943c0/screenshots/18.png)
![Further visualization](https://github.com/AnshPandey74/Prodigy_DS_03/raw/c8c4a73f890c3735a79817b8b8c6fb5b080943c0/screenshots/19.png)
![Further visualization](https://github.com/AnshPandey74/Prodigy_DS_03/raw/c8c4a73f890c3735a79817b8b8c6fb5b080943c0/screenshots/20.png)
![Further visualization](https://github.com/AnshPandey74/Prodigy_DS_03/raw/c8c4a73f890c3735a79817b8b8c6fb5b080943c0/screenshots/21.png)
![Further visualization](https://github.com/AnshPandey74/Prodigy_DS_03/raw/c8c4a73f890c3735a79817b8b8c6fb5b080943c0/screenshots/22.png)


## References
- UCI Machine Learning Repository: [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- Scikit-learn Documentation: [Decision Tree Classifier](https://scikit-learn.org/stable/modules/tree.html#classification)

Feel free to contribute to the project or provide feedback. For any issues or suggestions, please open an issue or submit a pull request.
