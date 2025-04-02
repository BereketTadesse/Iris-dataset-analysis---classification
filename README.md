# 🌸 Iris Flower Classification with Model Selection and Cross-Validation

This project presents a machine learning approach to classifying Iris flower species using Scikit-learn. It focuses on building multiple classification models, evaluating them using **cross-validation**, tuning hyperparameters with **RandomizedSearchCV**, and selecting the most accurate model based on consistent performance.

---

## 📂 Dataset Overview

- **Dataset**: Iris flower dataset (from Kaggle)
- **Samples**: 150
- **Classes**: Setosa, Versicolor, Virginica
- **Features**:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width

> **Note**: Correlation analysis revealed that **Petal Length** and **Petal Width** are the most important features for predicting the species.

---

## ✅ Why Cross-Validation?

Instead of using `train_test_split()` (which produces slightly different accuracy values each time due to randomness), this project uses **`cross_val_score()` with 5-fold cross-validation** to evaluate model performance.

### ✨ Advantages of Cross-Validation:
- Provides a **stable and reliable accuracy** by averaging performance over multiple splits.
- Ensures that every data point is used for both training and testing.
- Eliminates the inconsistency caused by random train/test splits.

---

## 🧠 Models Evaluated

The following models were trained and evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

Each model was evaluated using **5-fold cross-validation**, and performance was compared based on the **mean accuracy score**.

---

## 🔧 Hyperparameter Tuning

To improve model performance, **RandomizedSearchCV** was used for hyperparameter tuning. This method samples combinations of parameters randomly, reducing computation while still finding effective combinations.

- **Models Tuned**: Logistic Regression, Decision Tree, Random Forest, and SVM
- **Evaluation Metric**: Accuracy
- **Search Method**: 5-fold cross-validation using `RandomizedSearchCV`

---

## 🏆 Model Selection

After evaluation and tuning, models were compared based on:

- Mean cross-validated accuracy
- Simplicity and interpretability
- Stability across different data splits

> The model with the highest and most stable performance was selected as the best-performing classifier.

---

## 📌 Key Insights

- **Cross-validation** gives a more trustworthy accuracy compared to a single train/test split.
- **Petal Length** and **Petal Width** are the most important features, based on the correlation matrix.
- **RandomizedSearchCV** efficiently improves model accuracy with minimal computation.
- SVM performs best when using `gamma='scale'` (instead of `auto`).

---

## 📈 How to Use

### 1. Requirements

- Python
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

### 2. Running the Notebook

Open the `Iris_flower.ipynb` file and run the cells in order. You will see:

- A heatmap of feature correlations
- Evaluation of 4 models using 5-fold cross-validation
- Hyperparameter tuning using `RandomizedSearchCV`
- A final comparison to choose the best model

---

## 🧾 Conclusion

This project demonstrates:
- Proper model evaluation using cross-validation
- Efficient hyperparameter tuning
- Selection of the most accurate model using objective metrics

The workflow ensures **reproducible**, **fair**, and **robust** model comparisons — ideal for small datasets like Iris.

