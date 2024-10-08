#!/usr/bin/env python
# coding: utf-8

# ##**Place where we tune the max_features**##

# In[ ]:


import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


# Load the dataset
df = pd.read_csv("boss_email_dataset_encode_change_after_filter.csv")

# Split the data first
X_train, X_test, y_train, y_test = train_test_split(df[['subject', 'body', 'sender']], df['is_phishing'], test_size=0.2, random_state=100)


# Define models to test
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Random Forest': RandomForestClassifier(random_state=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=100),
    'Naive Bayes': MultinomialNB()
}

# Parameter grid for max_features
max_features_grid = [1000, 1500, 2000]

# Function to run grid search on both "subject" and "body" for all models
def run_grid_search(models, X_train, y_train):
    results = []
    
    for model_name, model in models.items():
        print(f"Running Grid Search for model: {model_name}")
        
        # Create a pipeline for the current model
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', model)
        ])
        
        # Define the parameter grid for max_features
        param_grid = {
            'tfidf__max_features': max_features_grid
        }
        
        # Grid search for the "subject" column
        grid_search_subject = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
        grid_search_subject.fit(X_train['subject'], y_train)
        
        # Store results for "subject"
        results.append({
            'model': model_name,
            'text_column': 'subject',
            'best_feature': grid_search_subject.best_params_,
            'best_score': grid_search_subject.best_score_
        })
        
        # Grid search for the "body" column
        grid_search_body = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
        grid_search_body.fit(X_train['body'], y_train)
        
        # Store results for "body"
        results.append({
            'model': model_name,
            'text_column': 'body',
            'best_feature': grid_search_body.best_params_,
            'best_score': grid_search_body.best_score_
        })
    
    # Print all results
    for result in results:
        print(f"\nModel: {result['model']} | Text Column: {result['text_column']}")
        print(f"Best max_features: {result['best_feature']['tfidf__max_features']}")
        print(f"Best accuracy score: {result['best_score']:.4f}")

# Run grid search across all models and both "subject" and "body" columns
run_grid_search(models, X_train, y_train)



# #Based on the 2000 max features that we selected to vectorize X-values, feed them into the ML models with default settings#

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Load the dataset
df = pd.read_csv("boss_email_dataset_encode_change_after_filter.csv")

# Split the data first
X_train, X_test, y_train, y_test = train_test_split(df[['subject', 'body', 'sender']], df['is_phishing'], test_size=0.2, random_state=100)

# Step 1: Vectorize the 'subject' column using TF-IDF, we do it seperately for test and train, to prevent data leakage, and prevent model from learning
tfidf_subject = TfidfVectorizer(max_features=2000, stop_words='english')
X_train_subject = tfidf_subject.fit_transform(X_train['subject']).toarray()
X_test_subject = tfidf_subject.transform(X_test['subject']).toarray()

# Step 2: Vectorize the 'body' column using TF-IDF
tfidf_body = TfidfVectorizer(max_features=2000, stop_words='english')
X_train_body = tfidf_body.fit_transform(X_train['body']).toarray()
X_test_body = tfidf_body.transform(X_test['body']).toarray()

# Step 3: Encode the 'sender' column using One-Hot Encoding, similarly, we do it seperately for test and train. If theres a new sender in testing set, the handle_unkown will ignore it
encoder = OneHotEncoder(handle_unknown='ignore')
X_train_sender = encoder.fit_transform(X_train[['sender']]).toarray()
X_test_sender = encoder.transform(X_test[['sender']]).toarray()

# Combine transformed features
X_train_combined = np.hstack([X_train_subject, X_train_body, X_train_sender])
X_test_combined = np.hstack([X_test_subject, X_test_body, X_test_sender])

def print_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix for {model_name}:\n", cm)

# ------------------------ Logistic regression ------------------------
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train_combined, y_train) #Where The model learns from the training set 
y_pred_log_reg = log_reg.predict(X_test_combined) #model predicts Y_test using the x_test set
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print(f"Logistic Regression Accuracy: {accuracy_log_reg:.2%}")
print_confusion_matrix(y_test, y_pred_log_reg, "Logistic Regression")

# ------------------------ Gradient Boosting ------------------------
gb_model = GradientBoostingClassifier(random_state=100)
gb_model.fit(X_train_combined, y_train)
y_pred_gb = gb_model.predict(X_test_combined)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"\nGradient Boosting Accuracy: {accuracy_gb:.2%}")
print_confusion_matrix(y_test, y_pred_gb, "Gradient Boosting")

# ------------------------ Random Forest ------------------------
rf_model = RandomForestClassifier(random_state=100)
rf_model.fit(X_train_combined, y_train)
y_pred_rf = rf_model.predict(X_test_combined)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nRandom Forest Accuracy: {accuracy_rf:.2%}")
print_confusion_matrix(y_test, y_pred_rf, "Random Forest")

# ------------------------ Naive Bayes ------------------------
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train_combined, y_train)
y_pred_nb = naive_bayes_model.predict(X_test_combined)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"\nNaive Bayes Accuracy: {accuracy_nb:.2%}")
print_confusion_matrix(y_test, y_pred_nb, "Naive Bayes")


# ##Further Hyperparameter tuning##

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Load the dataset
df = pd.read_csv("boss_email_dataset_encode_change_after_filter.csv")

# Split the data first
X_train, X_test, y_train, y_test = train_test_split(df[['subject', 'body', 'sender']], df['is_phishing'], test_size=0.2, random_state=100)

# Step 1: Vectorize the 'subject' column using TF-IDF
print("Starting TF-IDF Vectorization of 'subject' column...")
tfidf_subject = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_subject = tfidf_subject.fit_transform(X_train['subject']).toarray()
X_test_subject = tfidf_subject.transform(X_test['subject']).toarray()

# Step 2: Vectorize the 'body' column using TF-IDF
print("Starting TF-IDF Vectorization of 'body' column...")
tfidf_body = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_body = tfidf_body.fit_transform(X_train['body']).toarray()
X_test_body = tfidf_body.transform(X_test['body']).toarray()

# Step 3: Encode the 'sender' column using One-Hot Encoding
print("Starting One-Hot Encoding of 'sender' column...")
encoder = OneHotEncoder(handle_unknown='ignore')
X_train_sender = encoder.fit_transform(X_train[['sender']]).toarray()
X_test_sender = encoder.transform(X_test[['sender']]).toarray()

# Combine transformed features
print("Combining all transformed features...")
X_train_combined = np.hstack([X_train_subject, X_train_body, X_train_sender])
X_test_combined = np.hstack([X_test_subject, X_test_body, X_test_sender])

def print_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix for {model_name}:\n", cm)

# ------------------------ Hyperparameter Tuning ------------------------

# Parameter grids for each model
param_grid_log_reg = {
    'C': [0.01, 0.1, 1, 10],  # Regularization strength
    'penalty': ['l2'],        # Use 'l1' or 'l2' if using solver='liblinear'
}

param_grid_rf = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [None, 10, 20],     # Maximum depth of tree
    'min_samples_split': [2, 5],     # Minimum samples to split a node
    'min_samples_leaf': [1, 2]       # Minimum samples at a leaf node
}

param_grid_gb = {
    'n_estimators': [50, 100],  # Number of boosting stages
    'learning_rate': [0.01, 0.1, 0.2],  # Learning rate for each stage
    'max_depth': [3, 5, 7]  # Maximum depth of each tree
}

param_grid_nb = {
    'alpha': [0.1, 0.5, 1.0],  # Smoothing parameter
    'fit_prior': [True, False]  # Learn class prior probabilities or not
}

# ------------------------ Logistic Regression ------------------------
print("Starting Grid Search for Logistic Regression...")
log_reg = LogisticRegression(max_iter=200, solver='liblinear')
grid_search_log_reg = GridSearchCV(log_reg, param_grid_log_reg, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search_log_reg.fit(X_train_combined, y_train)
best_log_reg = grid_search_log_reg.best_estimator_

y_pred_log_reg = best_log_reg.predict(X_test_combined)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print(f"Best Hyperparameters for Logistic Regression: {grid_search_log_reg.best_params_}")
print(f"Logistic Regression Accuracy (Tuned): {accuracy_log_reg:.2%}")
print_confusion_matrix(y_test, y_pred_log_reg, "Logistic Regression")

# ------------------------ Gradient Boosting ------------------------
print("Starting Grid Search for Gradient Boosting...")
gb_model = GradientBoostingClassifier(random_state=100)
grid_search_gb = GridSearchCV(gb_model, param_grid_gb, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search_gb.fit(X_train_combined, y_train)
best_gb_model = grid_search_gb.best_estimator_

y_pred_gb = best_gb_model.predict(X_test_combined)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"\nGradient Boosting Accuracy (Tuned): {accuracy_gb:.2%}")
print_confusion_matrix(y_test, y_pred_gb, "Gradient Boosting")

# ------------------------ Random Forest ------------------------
print("Starting Grid Search for Random Forest...")
rf_model = RandomForestClassifier(random_state=100)
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search_rf.fit(X_train_combined, y_train)
best_rf_model = grid_search_rf.best_estimator_

y_pred_rf = best_rf_model.predict(X_test_combined)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nRandom Forest Accuracy (Tuned): {accuracy_rf:.2%}")
print_confusion_matrix(y_test, y_pred_rf, "Random Forest")

# ------------------------ Naive Bayes ------------------------
print("Starting Grid Search for Naive Bayes...")
naive_bayes_model = MultinomialNB()
grid_search_nb = GridSearchCV(naive_bayes_model, param_grid_nb, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search_nb.fit(X_train_combined, y_train)
best_nb_model = grid_search_nb.best_estimator_

y_pred_nb = best_nb_model.predict(X_test_combined)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"\nNaive Bayes Accuracy (Tuned): {accuracy_nb:.2%}")
print_confusion_matrix(y_test, y_pred_nb, "Naive Bayes")


# In[ ]:




