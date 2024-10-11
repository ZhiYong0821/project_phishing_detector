import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Load data
data = pd.read_csv("C:\\INF1002\\PhishingEmailProject\\emails.csv")

# Data preprocessing
data.drop_duplicates(inplace=True)
data['is_phishing'] = data['is_phishing'].replace({0: 'Not Phishing', 1: 'Phishing'})

# Combine subject, body, and from columns into a single message
data['message'] = data['subject'].fillna('') + " " + data['body'].fillna('') + " " + data['from'].fillna('')

# Drop any rows where 'message' is still empty (if needed)
data.dropna(subset=['message'], inplace=True)

# Prepare message and category
message = data['message']
category = data['is_phishing']

# Train-test split
(message_train, message_test, category_train, category_test) = train_test_split(message, category, test_size=0.2)

# Feature extraction
cv = CountVectorizer(stop_words="english")
features_train = cv.fit_transform(message_train)
features_test = cv.transform(message_test)

# Scale the features
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train.toarray())
features_test = scaler.transform(features_test.toarray())

# Creating and training the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(features_train, category_train)

# Predict Data
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result

# Streamlit app
st.header('Spam Detection')
input_message = st.text_area('Enter Message Here', height=150)

if st.button('Validate'):
    output = predict(input_message)
    try:
        result = str(output[0])
        if result == 'Phishing':
            st.write("Warning: This is a phishing email!")
        else:
            st.write("This is a legitimate email.")
    except Exception as e:
        st.write("An error occurred: " + str(e))
