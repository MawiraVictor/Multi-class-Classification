# Multi-class Classification

A machine learning project demonstrating multi-class classification strategies using logistic regression on the Obesity Risk Prediction dataset. The project compares One-vs-All (OvA) and One-vs-One (OvO) classification approaches.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [License](#license)
- [Author](#author)

---

## Overview

This project explores multi-class classification techniques applied to a real-world health dataset. The primary objectives are:

1. Understanding one-hot encoding for categorical variables
2. Implementing logistic regression for multi-class classification using One-vs-All and One-vs-One strategies
3. Evaluating model performance using appropriate metrics

The project demonstrates how different classification strategies impact model accuracy on the same dataset.

---

## Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)

**License:** [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)

**Size:** 2,111 samples with 17 attributes

### Features

| Variable | Type | Description |
|----------|------|-------------|
| Gender | Categorical | Gender of the individual |
| Age | Continuous | Age in years |
| Height | Continuous | Height in meters |
| Weight | Continuous | Weight in kilograms |
| family_history_with_overweight | Binary | Family history of overweight |
| FAVC | Binary | Frequent consumption of high caloric food |
| FCVC | Integer | Frequency of vegetable consumption |
| NCP | Continuous | Number of main meals daily |
| CAEC | Categorical | Food consumption between meals |
| SMOKE | Binary | Smoking status |
| CH2O | Continuous | Daily water consumption |
| SCC | Binary | Calorie monitoring |
| FAF | Continuous | Physical activity frequency |
| TUE | Integer | Technology device usage time |
| CALC | Categorical | Alcohol consumption frequency |
| MTRANS | Categorical | Primary mode of transportation |
| NObeyesdad | Categorical | Obesity level (target variable) |

### Target Classes

The target variable `NObeyesdad` contains seven obesity levels:

- Insufficient Weight
- Normal Weight
- Overweight Level I
- Overweight Level II
- Obesity Type I
- Obesity Type II
- Obesity Type III

The dataset is fairly balanced across all classes, eliminating the need for special handling of class imbalance.

---

## Project Structure

```
.
├── Multi-class Classification.ipynb          # Working notebook with implementation
├── Multi-class_Classification.ipynb          # Complete lab with exercises and solutions
└── README.md
```

### Notebook Contents

**Multi-class Classification.ipynb**
- Library installation and imports
- Data loading and exploration
- Feature scaling and one-hot encoding
- Target variable encoding
- Model training (OvA and OvO)
- Performance evaluation

**Multi-class_Classification.ipynb**
- Complete structured lab format
- Dataset documentation
- Exercise solutions
- Additional analysis and visualizations

---

## Requirements

- Python 3.10 or higher
- NumPy 2.2.0
- Pandas 2.2.3
- scikit-learn 1.6.0
- Matplotlib 3.9.3
- Seaborn 0.13.2

---

## Installation

1. Clone the repository or download the notebook files.

2. Install the required dependencies:

```bash
pip install numpy==2.2.0 pandas==2.2.3 scikit-learn==1.6.0 matplotlib==3.9.3 seaborn==0.13.2
```

3. Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

4. Open either notebook file to explore the implementation.

---

## Usage

### Data Preprocessing

The preprocessing pipeline consists of three main steps:

**Feature Scaling**

Continuous numerical features are standardized using `StandardScaler` to ensure all features contribute equally to model training:

```python
continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[continuous_columns])
```

**One-Hot Encoding**

Categorical variables are converted to numerical format using `OneHotEncoder` with the first category dropped to avoid multicollinearity:

```python
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])
```

**Target Encoding**

The target variable is converted to integer codes:

```python
prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes
```

### Model Training

**One-vs-All Strategy**

```python
model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ova.fit(X_train, y_train)
```

**One-vs-One Strategy**

```python
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(X_train, y_train)
```

### Evaluation

```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

---

## Methodology

### One-vs-All (OvA)

The One-vs-All approach trains a single binary classifier for each class. Each classifier learns to distinguish one class from all other classes combined. For k classes, k classifiers are trained. During prediction, the class with the highest confidence score is selected.

**Advantages:**
- Simpler implementation with fewer classifiers
- Efficient for algorithms that naturally provide confidence scores

**Disadvantages:**
- Classifiers may struggle with class imbalance
- Each classifier must perform well with highly imbalanced data

### One-vs-One (OvO)

The One-vs-One approach trains a binary classifier for every pair of classes. For k classes, this results in k(k-1)/2 classifiers. During prediction, a voting mechanism determines the final class based on majority wins.

**Advantages:**
- Suitable for computationally expensive algorithms
- Each classifier deals with smaller, more balanced datasets
- Can achieve higher accuracy in certain scenarios

**Disadvantages:**
- Computationally expensive for datasets with many classes
- Potential for ambiguous predictions in case of voting ties

---

## Results

### Model Comparison

| Strategy | Accuracy |
|----------|----------|
| One-vs-All (OvA) | 76.12% |
| One-vs-One (OvO) | 92.20% |

The One-vs-One strategy significantly outperforms One-vs-All on this dataset. This improvement can be attributed to:

1. **Smaller binary problems:** Each classifier focuses on distinguishing between only two classes, leading to more specialized decision boundaries.

2. **Better handling of class overlap:** With seven classes, some classes may have overlapping feature distributions. OvO classifiers can better capture the nuances between specific pairs.

3. **Balanced training data:** Each binary classifier uses only samples from two classes, which are more balanced compared to the OvA approach where one class is compared against all others combined.

### Key Observations

- The dataset is well-balanced across all seven obesity levels
- Standardization of continuous features is essential for logistic regression performance
- One-hot encoding with `drop='first'` prevents multicollinearity while maintaining model interpretability
- The OvO strategy is computationally more expensive but yields superior accuracy for this multi-class problem

---

## Exercises

The complete lab notebook includes the following exercises with solutions:

1. **Test Size Experimentation:** Vary the train-test split ratio (0.1, 0.3) and observe performance impact.

2. **Feature Importance Visualization:** Plot bar charts showing feature importance based on model coefficients for both OvA and OvO strategies.

3. **Pipeline Automation:** Implement a function to automate the entire workflow from data loading to model evaluation.

---

## License

This project uses data licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/legalcode).

The code is provided for educational purposes.

---

**Organization:** IBM Corporation

---

## Acknowledgments

- UCI Machine Learning Repository for providing the Obesity Risk Prediction dataset
- scikit-learn documentation and community for implementation guidance
- IBM Skills Network for the lab structure and educational framework
