import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('email_dataset.csv')
df['text'] = df['subject'] + ' ' + df['body']

# Handle NaN values by replacing them with an empty string
df['text'] = df['text'].fillna('')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['is_phishing'], test_size=0.2, random_state=42)

# Feature extraction 
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train and evaluate models
models = {
    'Logistics Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier()
}

for name, model in models.items():
    model.fit(X_train_vec, y_train)  # Fixed the method typo here
    y_pred = model.predict(X_test_vec)

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
    
print("Model training and evaluation completed")