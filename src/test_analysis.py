import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from nltk.corpus import stopwords
import nltk

# Ensure that you have NLTK stopwords
nltk.download('stopwords')

# Load data
data = pd.read_csv("/Users/zhiyong/Downloads/emails.csv")

# Data preprocessing
data.drop_duplicates(inplace=True)
data['is_phishing'] = data['is_phishing'].replace({0: 'Not Phishing', 1: 'Phishing'})

# Combine subject, body, and from columns into a single message
data['message'] = data['subject'].fillna('') + " " + data['body'].fillna('') + " " + data['from'].fillna('')

# Drop any rows where 'message' is still empty
data.dropna(subset=['message'], inplace=True)

# Remove HTML tags from messages
data['message'] = data['message'].apply(lambda x: re.sub(r'<.*?>', '', x))

# Train-test split
message = data['message']
category = data['is_phishing']
message_train, message_test, category_train, category_test = train_test_split(message, category, test_size=0.2, random_state=42)

# Feature extraction
cv = CountVectorizer(stop_words="english", max_features=5000)
features_train = cv.fit_transform(message_train)
features_test = cv.transform(message_test)

# Scale the features
scaler = StandardScaler(with_mean=False)  # Don't center sparse matrices
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

# Hyperparameter tuning with Cross Validation
param_grid = {
    'C': [0.1, 1, 10],  # Regularization strength
    'penalty': ['l2'],
    'solver': ['liblinear'],
    'max_iter': [100, 200]
}

model = LogisticRegression()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(features_train, category_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Predict Data with probability
def predict(message):
    input_message = cv.transform([message])
    input_message = scaler.transform(input_message)
    result = best_model.predict(input_message)
    prob = best_model.predict_proba(input_message)
    return result, prob

# Load unwanted terms from a file
def load_unwanted_terms(file_path):
    try:
        with open(file_path, 'r') as file:
            return {line.strip().lower() for line in file if line.strip()}
    except FileNotFoundError:
        print(f"File not found: {file_path}, proceeding without unwanted terms.")
        return set()

# Load unwanted terms
unwanted_terms = load_unwanted_terms('Deploy_+_Visual/UnwantedTerms.txt')

# Combine unwanted terms and stop words
stop_words = list(set(unwanted_terms).union(set(stopwords.words('english'))))

# Filter valid English words from a message
def filter_valid_words(text):
    return re.findall(r'\b[a-zA-Z]+\b', text)

# Class distribution plot
def plot_class_distribution(data, filename):
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='is_phishing', data=data, palette='Blues', hue='is_phishing', dodge=False)
    plt.title('Distribution of Phishing vs Legitimate Emails')
    plt.xlabel('Email Classification')
    plt.ylabel('Count')

    # Add the counts on top of each bar
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Plot Word Count Distribution
def plot_word_count_distribution(data, filename):
    data['word_count'] = data['message'].apply(lambda x: len(x.split()))
    
    plt.figure(figsize=(8, 6))
    sns.histplot(data=data, x='word_count', hue='is_phishing', kde=False, palette='Set2')
    plt.title('Word Count Distribution (Phishing vs Legitimate)')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.legend(title='Email Type', labels=['Legitimate Emails', 'Phishing Emails'])
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Plot Special Character Distribution
def plot_special_char_distribution(data, filename):
    data['special_char_count'] = data['message'].apply(lambda x: sum([1 for char in x if char in '!$%']))
    
    valid_data = data[data['special_char_count'] > 0]  # Only show emails with special characters
    
    plt.figure(figsize=(8, 6))
    sns.histplot(data=valid_data, x='special_char_count', hue='is_phishing', kde=False, palette='Set1')
    plt.title('Special Character Usage (Phishing vs Legitimate)')
    plt.xlabel('Special Characters Count')
    plt.ylabel('Frequency')
    plt.legend(title='Email Type', labels=['Legitimate Emails', 'Phishing Emails'])
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Generate TF-IDF vectorization and bar plot with unwanted terms excluded
def plot_common_words(data, title, filename, stop_words):
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=20)
    word_count = vectorizer.fit_transform(data)
    word_freq_array = word_count.toarray().sum(axis=0)
    
    word_freq = pd.DataFrame({'tfidf_score': word_freq_array}, index=vectorizer.get_feature_names_out())
    word_freq = word_freq.sort_values('tfidf_score', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=word_freq.index, y='tfidf_score', data=word_freq, palette='Blues_r')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Generate word cloud with unwanted terms excluded
def generate_wordcloud(data, title, filename, stop_words):
    text = ' '.join(data)
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Reds',
                          stopwords=stop_words, max_words=100, contour_width=3, contour_color='black').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, pad=20)
    plt.savefig(filename)
    plt.show()

# Apply to phishing and legitimate emails
phishing_emails = data[data['is_phishing'] == 'Phishing']['message']
legitimate_emails = data[data['is_phishing'] == 'Not Phishing']['message']

# Plot all graphs
plot_class_distribution(data, 'class_distribution2.png')
plot_word_count_distribution(data, 'word_count_distribution2.png')
plot_special_char_distribution(data, 'special_char_distribution2.png')

plot_common_words(phishing_emails, 'Top 20 Words in Phishing Emails', 'phishing_words2.png', stop_words)
plot_common_words(legitimate_emails, 'Top 20 Words in Legitimate Emails', 'legitimate_words2.png', stop_words)

generate_wordcloud(phishing_emails, 'Words to Be Aware Of in Phishing Emails', 'phishing_wordcloud2.png', stop_words)
generate_wordcloud(legitimate_emails, 'Words Commonly Used in Legitimate Emails', 'legitimate_wordcloud2.png', stop_words)

print(r'Successful Data Analysis. Visualizations saved under "..Project Phishing Detector\src" .')