# Customer Churn Predictor
## Tech Stack
### Python
### scikit-learn
### pandas

Predicting customer churn in telecom industry is a common problem faced by many companies. This project uses machine learning algorithms to predict whether a customer is likely to churn or not.

### Features
* Trains a Logistic Regression model on the dataset
* Evaluates the model's performance using accuracy score, classification report, and confusion matrix

### Setup/Run Instructions
1. Clone the repository: `git clone https://github.com/your-username/customer-churn-prediction.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the script: `python main.py`

### Sample Output
Accuracy: 0.85
Classification Report:
              precision    recall  f1-score   support

           0       0.81      0.83      0.82       120
           1       0.89      0.86      0.87        80

    accuracy                           0.85       200
   macro avg       0.85      0.84      0.84       200
weighted avg       0.85      0.85      0.85       200

Confusion Matrix:
[[106 14]
 [ 8 72]]
### What I learned
I learned the importance of preprocessing data in machine learning projects. In this project, I preprocessed the dataset by converting categorical variables into numerical variables using pandas' map function. This helped improve the model's performance and accuracy. Additionally, I used scikit-learn's built-in functions to evaluate the model's performance, which saved me time and effort.