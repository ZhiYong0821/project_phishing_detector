import os
import pandas as pd
import email
import chardet

def extract_email_features(filepath):
    # Detect the file encoding
    with open(filepath, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        file_encoding = result['encoding']

    # Open and read email file with detected encoding, fallback to utf-8
    try:
        with open(filepath, 'r', encoding=file_encoding) as file:
            content = file.read()
    except UnicodeDecodeError:
        # If detection fails, try with utf-8
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()

    # Parse email content
    msg = email.message_from_string(content)
    
    # Read the email content and retrieve relevant parts of the email such as subject, sender, and body
    subject = msg['subject'] or ''
    sender = msg['from'] or ''
    
    # Handle multipart messages and potential encoding issues in the body
    if msg.is_multipart():
        body = ''
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    body += part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', errors='ignore')
                except:
                    body += part.get_payload(decode=False)
    else:
        try:
            body = msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8', errors='ignore')
        except:
            body = msg.get_payload(decode=False)

    return {
        'subject': subject,
        'from': sender,
        'body': body,
    }

# prepares dataset from all of the email files
def prepare_dataset(base_directory):
    data = []
    # defines categories according to email type / labels
    categories = {
        'easy_ham': 0, # legitimate emails (easy to tell)
        'hard_ham': 0, # legitimate emails (harder to tell but still are legit emails)
        'spam': 1 # Confirmed phishing/spam emails
    }

    # run through each category of the email dataset
    for category, is_phishing in categories.items():
        directory = os.path.join(base_directory, category)
        # process and join each file in each email category directory
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                # extract relevant features from the emails
                features = extract_email_features(filepath)
                features['is_phishing'] = is_phishing # adds the label accordingly
                data.append(features)

    # returns the list of email (in form of dictionaries) to a dataframe
    return pd.DataFrame(data)

base_dir = r"C:\Users\Admin\Documents\Code\School Assignments\Project Phishing Detector\src\datasets" 
#base_dir = '/Users/zhiyong/project_phishing_detector/src/datasets'

#assigns the list of email dictionaries to df variable
df = prepare_dataset(base_dir)

df.dropna()
#df_clean = df.dropna()

# Optionally check again for any remaining missing values
print("Missing values in each column:\n", df.isnull().sum())

# converts prepared dictionary dataset to a CSV file
df.to_csv('email_dataset_encode_change_final.csv', index=False, encoding='utf-8', na_rep='')

#print(f"Dataset prepared with {len(df)} emails")