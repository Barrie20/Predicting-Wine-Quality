# Wine Quality Prediction using Multiple Linear Regression

## Introduction

This project aims to predict the quality of white wine using a dataset containing various chemical properties of the wine. The quality of wine is rated on a scale from 0 to 10. We employ Multiple Linear Regression to build the predictive model. The project involves data preprocessing, training, evaluating, and analyzing the model.

## Dataset

The dataset consists of the following features:
- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol
- Quality (label)

The dataset can be found [here](winequality-white.csv).

## Steps to Follow

1. **Exploring and Identifying the Problem**:
   - Load the dataset and display the first few rows.
   - Check for missing data.

2. **Applying Appropriate Machine Learning Algorithm**:
   - Multiple Linear Regression is chosen for this task.

3. **Implementing the Solution**:
   - Perform exploratory data analysis (EDA).
   - Visualize the data distribution and outliers using box plots.
   - Analyze the correlation between features using a heatmap.
   - Split the data into training and testing sets.
   - Scale the features.
   - Train the model using Multiple Linear Regression.

4. **Evaluating and Improving the Model**:
   - Evaluate the model using the R² score.
   - Visualize the predicted quality versus the true quality.

5. **Analyzing the Result**:
   - Scatter plot to visualize the relationship between true and predicted quality.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/wine-quality-prediction.git
   cd wine-quality-prediction
   ```

2. Install the required packages:
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn missingno
   ```

## Usage

1. Load and explore the dataset:
   ```python
   wine_df = pd.read_csv('winequality-white.csv', sep=';')
   wine_df.head()
   ```

2. Check for missing data:
   ```python
   msno.matrix(wine_df, figsize=(10, 3))
   ```

3. Visualize data distribution and outliers:
   ```python
   fig, axes = plt.subplots(nrows=2, ncols=1)
   fig.set_size_inches(20, 30)
   sns.boxplot(data=wine_df, orient="v", ax=axes[0])
   sns.boxplot(data=wine_df, y="quality", orient="pH", ax=axes[1])
   ```

4. Analyze feature correlation:
   ```python
   corr_mat = wine_df.corr()
   mask = np.array(corr_mat)
   mask[np.tril_indices_from(mask)] = False
   fig, ax = plt.subplots()
   fig.set_size_inches(20, 10)
   sns.heatmap(corr_mat, mask=mask, vmax=0.8, square=True, annot=True)
   ```

5. Split and scale the data:
   ```python
   X = wine_df.iloc[:, :-1]
   y = wine_df.iloc[:, -1]
   X = np.append(arr=np.ones((X.shape[0], 1)), values=X, axis=1)
   X_train, X_test, y_train, y_test = train_test_split(X, y)
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test = scaler.fit_transform(X_test)
   ```

6. Train the model and make predictions:
   ```python
   lr = LinearRegression()
   lr.fit(X_train, y_train)
   y_pred = lr.predict(X_test)
   ```

7. Evaluate the model:
   ```python
   from sklearn.metrics import r2_score
   r2_score(y_test, y_pred)
   ```

8. Visualize the results:
   ```python
   plt.scatter(y_test, y_pred, c='g')
   plt.xlabel('True Quality')
   plt.ylabel('Predicted Quality')
   plt.title('Predicted quality vs True quality')
   plt.show()
   ```

## Results

- The R² score of the model: 0.2657547922902175
- Scatter plot of predicted quality vs. true quality shows the relationship between the predictions and actual values.

## Conclusion

The Multiple Linear Regression model provides a moderate fit for predicting wine quality based on its chemical properties. Further improvements can be made by experimenting with other algorithms, feature engineering, or parameter tuning.
