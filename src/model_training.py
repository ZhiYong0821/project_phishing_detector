import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load the data
df = pd.read_csv('boss_email_dataset_encode_change_after_filter.csv') # Updated the file name - Yong Ying
df['text'] = df['subject'] + ' ' + df['body']

# Handle NaN values by replacing them with an empty string
df['text'] = df['text'].fillna('')
df['body'] = df['body'].fillna('')

# Added function to extract links / urls from email body - Yong Ying
def url_extraction(text):
    # Using regular expressions to find urls 
    url_pattern = r'(https?://[^\s]+|www\.[^\s]+)'
    urls_find = re.findall(url_pattern, text)
    return urls_find

df['num_urls'] = df['body'].apply(lambda x: len(url_extraction(x)))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df[['text', 'num_urls']], df['is_phishing'], test_size=0.2, random_state=42)

# Feature extraction 
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train['text'])
X_test_vec = vectorizer.transform(X_test['text'])

# Convert the 'numUrls' column to sparse format and combine with TF-IDF features 
from scipy.sparse import hstack

X_train_combined = hstack([X_train_vec, X_train[['num_urls']].values])
X_test_combined = hstack([X_test_vec, X_test[['num_urls']].values])

# Train and evaluate models
models = {
    'Logistics Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier()
}

for name, model in models.items():
    model.fit(X_train_combined, y_train)  # Fixed the method typo here
    y_pred = model.predict(X_test_combined)

    print(f"\n{name} Results:")
    print(classification_report(y_test, y_pred))

    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{name.lower().replace(" ", "_")}_confusion_matrix.png')

# To print out the number of urls summary - Yong Ying
url_data = df['num_urls'].value_counts()

summary_email = {
    'Total Emails': len(df),
    'Emails with URLs': df[df['num_urls'] > 0].shape[0],
    'Emails with 1 URL': df[df['num_urls'] == 1].shape[0],
    'Emails with 2+ URLs': df[df['num_urls'] > 1].shape[0],
    'Emails with 0 URLs': df[df['num_urls'] == 0].shape[0]
}

for key, value in summary_email.items():
    print(f'{key}: {value}')


# This code is to remove any NaN values and also add num_url column to excel file 
df.dropna()

df.to_csv("boss_email_dataset_encode_change_after_filter.csv", index=False, encoding='utf-8', na_rep='')

print("Model training and feature extraction completed")