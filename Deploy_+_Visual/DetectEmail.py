import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

# Load data
data = pd.read_csv("../src/boss_email_dataset_encode_change_after_filter.csv")

# Data preprocessing
data.drop_duplicates(inplace=True)
data['is_phishing'] = data['is_phishing'].replace({0: 'Not Phishing', 1: 'Phishing'})

# Combine subject, body, and from columns into a single message
data['message'] = data['subject'].fillna('') + " " + data['body'].fillna('') + " " + data['sender'].fillna('')

# Drop any rows where 'message' is still empty
data.dropna(subset=['message'], inplace=True)

# Remove HTML tags from messages
data['message'] = data['message'].apply(lambda x: re.sub(r'<.*?>', '', x))

# Prepare message and category
message = data['message']
category = data['is_phishing']

# Train-test split
message_train, message_test, category_train, category_test = train_test_split(message, category, test_size=0.2, random_state=42)

# Feature extraction
cv = CountVectorizer(stop_words="english", max_features=5000)  # Limit features to reduce overfitting
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

# Track user statistics of whether it is legitimate or phishing
if 'phishing_count' not in st.session_state:
    st.session_state['phishing_count'] = 0
if 'legitimate_count' not in st.session_state:
    st.session_state['legitimate_count'] = 0

# Predict Data with probability
def predict(message):
    input_message = cv.transform([message])
    input_message = scaler.transform(input_message)
    result = best_model.predict(input_message)
    prob = best_model.predict_proba(input_message)
    return result, prob

# Load unwanted terms from a file
def load_unwanted_terms(file_path):
    with open(file_path, 'r') as file:
        # Read lines and strip whitespace
        return {line.strip().lower() for line in file if line.strip()}

# Load unwanted terms
unwanted_terms = load_unwanted_terms('UnwantedTerms.txt')

# Filter valid English words from a message
def filter_valid_words(text):
    # Use a simple regex to find valid words (consisting of alphabetical characters)
    return re.findall(r'\b[a-zA-Z]+\b', text)

# Streamlit app
st.header('Phishing Email Detection')

# User Input
input_message = st.text_area('Enter Message Here', height=150)

if st.button('Validate'):
    # Make prediction and get the probability
    output, prob = predict(input_message)
    
    # Display prediction result
    result = str(output[0])
    confidence = prob[0][1] if result == 'Phishing' else prob[0][0]
    confidence_percent = int(confidence * 100)
    
    # Update statistics
    if result == 'Phishing':
        st.session_state['phishing_count'] += 1
        st.error("Warning: This is a phishing email!")
    else:
        st.session_state['legitimate_count'] += 1
        st.success("This is a legitimate email.")
    
    # Show certainty level with a progress bar
    st.write("What is the prediction confidence for this result (in percentage)?")
    # Output the confidence percentage
    st.write(f"The prediction confidence is **{confidence_percent}%**.")
    
    # Create progress bar with 0% and 100% labels and show the prediction percentage
    fig, ax = plt.subplots(figsize=(8, 1))
    
    # Draw a horizontal line from 0% to 100%
    ax.plot([0, 1], [0, 0], color='blue', lw=3)
    
    # Add 'O' markers for 0% and 100%
    ax.plot(0, 0, marker='o', markersize=5, color='black')  # Marker for 0%
    ax.plot(1, 0, marker='o', markersize=5, color='black')  # Marker for 100%
    # Add labels for 0% and 100%
    ax.text(0, -0.02, '0%', horizontalalignment='center', verticalalignment='center', fontsize=12)
    ax.text(1, -0.02, '100%', horizontalalignment='center', verticalalignment='center', fontsize=12)
    
    # Adjust the confidence label position to prevent overlap and make it visually appealing
    ax.text(confidence, 0.02, f'{confidence_percent}%', horizontalalignment='center', verticalalignment='bottom', fontsize=12, color='red')
    ax.plot([confidence], [0], marker='|', markersize=10, color='red')
    
    ax.set_xlim(-0.05,1.05)
    ax.set_axis_off()
    st.pyplot(fig)

    st.subheader("User's Input Count (Phishing vs. Legitimate):")
    # Count phishing and legitimate emails from the dataset
    phishing_count = data[data['is_phishing'] == 'Phishing'].shape[0]
    legitimate_count = data[data['is_phishing'] == 'Not Phishing'].shape[0]

    # Create bar chart for user statistics
    fig, ax = plt.subplots()
    ax.bar(['Phishing Emails', 'Legitimate Emails'], [st.session_state['phishing_count'], st.session_state['legitimate_count']], color=['red', 'green'])
    ax.set_ylabel('Number of Emails')
    ax.set_yticks(range(0, max(st.session_state['phishing_count'], st.session_state['legitimate_count']) + 1))
    ax.set_title('Phishing vs. Legitimate Emails Detected')
    st.pyplot(fig)

    # Side-by-side layout: Pie chart and Word Cloud based on past dataset 
    st.subheader("These visualizations below are based on past dataset.")
    col1, col2 = st.columns(2)

    with col1:
        # Generate data for the overall pie chart showing the distribution of phishing and legitimate emails in the dataset
        phishing_count_total = data[data['is_phishing'] == 'Phishing'].shape[0]
        legitimate_count_total = data[data['is_phishing'] == 'Not Phishing'].shape[0]

        # Prepare data for the pie chart
        categories = ['Phishing Emails', 'Legitimate Emails']
        counts = [phishing_count_total, legitimate_count_total]

        # Create a pie chart for the overall dataset
        fig, ax = plt.subplots()
        ax.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90, colors=['red', 'green'])
        ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
        st.write("Distribution of Phishing vs. Legitimate Emails in the dataset")
        st.pyplot(fig)

    with col2:
        # Generate a word cloud based on the entire dataset of messages marked as phishing
        phishing_messages = data[data['is_phishing'] == 'Phishing']['message']
        wordcloud_data = ' '.join(phishing_messages)
        
        # Filter out only valid English words
        filtered_words = filter_valid_words(wordcloud_data)

        # Create a list for unwanted terms found in the text
        found_unwanted_terms = [word for word in filtered_words if word.lower() in unwanted_terms]

        # Remove unwanted terms from the filtered words
        filtered_words = [word for word in filtered_words if word.lower() not in unwanted_terms]
        wordcloud_data_filtered = ' '.join(filtered_words)

        # Create and display the word cloud
        wordcloud = WordCloud(width=400, height=400, background_color='white', colormap='Reds').generate(wordcloud_data_filtered)
        st.image(wordcloud.to_array(), use_column_width=True,caption='Words to Be Aware Of in Phishing Emails')
