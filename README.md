# Multiple Myeloma Survival Prediction
This repository contains a solution to the [Multiple Myeloma Survival](https://kaggle.com/competitions/multiple-myeloma-survival) competition in Kaggle, where the goal is to predict the survival time of patients with multiple myeloma. This competition was part of the Machine Learning (2023/24) course at FCT-UNL.


## Dataset Description
### Columns
#### Features:
- **Age**: Age of the patient (integer).
- **Gender**: Should be understood as biological sex (binary: Male/Female).
- **Stage**: Stage of the cancer (ordinal: 1-4).
- **GeneticRisk**: A real number between 0 and 1 representing genetic cancer risk.
- **TreatmentType**: Type of treatment (binary: Aggressive/Non-aggressive).
- **ComorbidityIndex**: A numerical score representing the severity and number of comorbid conditions.
- **TreatmentResponse**: Effectiveness of the treatment (binary: Poor/Good).


#### Label and Censoring:
- **SurvivalTime**: Time from the start of the study until the event of interest (continuous).
- **Censored**: Censoring indicator (binary: 1 if censored, 0 if the event is observed).

### Files
- **train_data.csv**: Training set containing features, survival times, and censoring indicators. Includes missing values.
- **test_data.csv**: Test set containing features but without survival times or censoring information (for prediction submission).

## Modeling Approach
The solution employs a semi-supervised learning approach to predict the survival time of patients with multiple myeloma. In this context, semi-supervised learning means that the model leverages both labeled and unlabeled data, to an extent. The unlabeled data is only included in the preprocessing phase, but not in the training of the model itself.

### Data Preprocessing
The data preprocessing included the following tasks:
- **Handling Missing Values**: Missing values in both features and survival times were imputed using various strategies. Most of them coming from the `sklearn.impute` module.
- **Scaling and Encoding**: Numerical features were standardized, while an ordinal encoding was applied to categorical features.
- **Dimensionality Reduction**: PCA was applied to reduce the dimensionality of the feature space.

### Model Selection
To evaluate a model's performance before submission, cross-validation was utilized.  The cross-validation technique used involved dividing the labeled dataset into a 
test set and another set, stratified on the `Censored` column. The latter set would 
then be used for stratified 5-fold cross-validation, also stratified on the `Censored`, due to the custom metric used for evaluation.

Multiple combinations of models and preprocessing techniques were tested. The final model was selected based on cross-validation performance and leaderboard results. The following models were experimented with:
- Linear Regression
- L2 Regularized Linear Regression (Ridge)
- K-Nearest Neighbors Regressor
- Decision Tree Regressor
- Random Forest Regressor
- Histogram-based Gradient Boosting Regressor
- SVR

In the end, the best performing models were the simpler ones: Linear Regression and L2 Regularized Linear Regression (Ridge).
    

## Evaluation
Models were evaluated on a custom metric named Censored Mean Squared Error (cMSE) between the predicted survival and the observed target. The cMSE is defined as:

$cMSE = \frac{1}{N} \sum_{n=1}^{N} \left[ (1 - c_n) \cdot (y_n - \hat{y}_n)^2 + c_n \cdot \max(0, y_n - \hat{y}_n)^2 \right]$

Where:

- N: Number of samples.

- c_n: Censoring indicator for sample n (1 if censored, 0 if not).

- y_n: True survival time for sample n.

- $\hat{y}_n$: Predicted survival time for sample n.


## Results

All the models are in the `models` folder. Each model is saved as a `.pkl` file and has a corresponding text file describing what its pipeline looks like. 

The final submitted predictions came from  `model_12.pkl`. This was the model that performed best on the public leaderboard and cross-validation ensured that it was not overfitting as it still performed well on the private leaderboard while most other participants' models were overfitting.

### Leaderboard Score
- **Public**: 1.22758 (14th place)
- **Private**: 1.32156 (1st place)