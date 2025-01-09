# Beyond the Waiting Room: Data-Driven Appointment Predictions

## Project Overview
This project analyzes a public dataset of medical appointment no-shows in Brazil, aiming to build a machine learning model that predicts whether a patient will show up for their scheduled appointment. By using gradient boosting and other machine learning techniques, this project seeks to help healthcare providers minimize no-show rates and optimize resource allocation.

## Dataset Description
### Source

The dataset is publicly available on Kaggle: Medical Appointment No Shows Dataset. [Here is one copy of the dataset.](https://www.kaggle.com/datasets/joniarroba/noshowappointments) 

| **Column Name**   | **Description**                                                                                   |
|------------------ |---------------------------------------------------------------------------------------------------|
| `patient_id`      | Unique identifier for each patient.                                                                |
| `appointment_id`  | Unique identifier for each appointment.                                                             |
| `scheduled_day`   | The date when the appointment was scheduled.                                                        |
| `appointment_day` | The date when the appointment was supposed to happen.                                               |
| `age`             | Age of the patient.                                                                                 |
| `neighbourhood`   | The location where the appointment took place.                                                       |
| `scholarship`     | Indicates whether the patient is enrolled in the Brazilian Bolsa Família social welfare program (1 = Yes, 0 = No). |
| `hypertension`    | Indicates whether the patient has hypertension (1 = Yes, 0 = No).                                   |
| `diabetes`        | Indicates whether the patient has diabetes (1 = Yes, 0 = No).                                       |
| `alcoholism`      | Indicates whether the patient has a history of alcoholism (1 = Yes, 0 = No).                        |
| `handicap`        | Indicates the number of disabilities (0 = None, 1-4 = varying severity levels).                     |
| `sms_received`    | Indicates whether the patient received a reminder SMS (1 = Yes, 0 = No).                            |
| `no_show`         | The target column indicating whether the patient showed up for the appointment ("Yes" = No-show, "No" = Showed up). |

## Context
The dataset contains information on 110,527 medical appointments from Brazil in 2016. Each row represents a patient’s appointment and includes several features about the patient and the appointment itself.

## Project Hypothesis
The primary hypothesis for this project is:<br>
Patients who face socioeconomic challenges, chronic conditions, or limited access to healthcare resources are more likely to miss their appointments.<br>
Reminder SMS messages may positively impact attendance, though the effect may be limited depending on other circumstances.

## Methodology
This project follows the CRISP-DM (Cross Industry Standard Process for Data Mining) methodology:

Business Understanding: Identify the research question—predicting no-shows to optimize appointment scheduling and minimize idle time.<br>
Data Understanding: Explore the dataset to understand distributions, missing data, and potential biases.<br>
Data Preparation: Clean the data by addressing missing values, encoding categorical columns (like gender), and engineering new features (e.g., waiting_time).<br>
Modeling: Train several machine learning models:<br>
Logistic Regression<br>
Random Forest Classifier<br>
Gradient Boosting Classifier (tuned for hyperparameters like n_estimators, max_depth, and learning_rate)<br>
Evaluation: Evaluate model performance using accuracy, precision, recall, and F1-score, with cross-validation.<br>
Deployment Plan: Save the trained model to make predictions and suggest improvements for real-world implementation.<br>

## Results

The Gradient Boosting Classifier achieved the highest accuracy and was tuned for hyperparameters (n_estimators=600, learning_rate, max_depth, and min_samples_split). Initial results included:

>Accuracy: 71.7%
>Precision: ~54%
>Recall: ~1.39% (before tuning, indicating room for improvement).
>
## Technologies Used
Python 3.10: Core language for analysis and modeling.
#### Libraries:
* pandas for data manipulation
* numpy for numerical operations
* scikit-learn for machine learning models and evaluation metrics
* matplotlib and seaborn for visualizations
* Jupyter Notebooks: Used for exploratory data analysis (EDA).
* Joblib: Used for saving and loading trained models.
* (future)GitHub Actions: For continuous integration (CI) testing.
## Project Deliverables
* Models: A tuned Gradient Boosting Classifier saved as tuned_gradient_boosting_model.pkl.
* Visualizations: Graphs illustrating data distribution, feature importance, and model performance.
* Reports: Documentation of findings and improvement suggestions.

## Installation and Usage

``` python
git clone https://github.com/username/repository-name.git
cd repository-name

#create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # MacOS/Linux
venv\Scripts\activate  # Windows

#install dependencies
pip install -r requirements.txt

#run the main script:
python main.py
```
## Future Improvements
Explore additional hyperparameters (subsample, max_features) to further tune the Gradient Boosting model.<br>
Investigate imbalanced class strategies, such as SMOTE (Synthetic Minority Oversampling Technique).<br>
Consider external data integration (e.g., public health records) to improve predictions.<br>

## Acknowledgments
The dataset was shared by many on Kaggle, including by Joni Hoppen.<br>
Thanks to Dr. Jim Ashe and Kelly Smith-IT and Prof. Larry Burdick of Western Governors University for your advice, encouragement and insights. <br>
Thanks to the wider healthcare research community for providing insight into appointment management challenges.
