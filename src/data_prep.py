import os
import pandas as pd
import email

def extract_email_features(filepath):
    with open(filepath, 'r', encoding='latin-1') as file:
        content = file.read()
    msg = email.message_from_string(content)
    return {
        'subject': msg['subject'] or '',
        'from': msg['from'] or '',
        'body': msg.get_payload() if msg.is_multipart() else msg.get_payload(),
    }

def prepare_dataset(base_directory):
    data = []
    categories = {
        'easy_ham': 0,
        'hard_ham': 0,
        'spam': 1
    }

    for category, is_phishing in categories.items():
        directory = os.path.join(base_directory, category)
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                features = extract_email_features(filepath)
                features['is_phishing'] = is_phishing
                data.append(features)

    return pd.DataFrame(data)

base_dir = r"C:\Users\Admin\Documents\Code\School Assignments\Project Phishing Detector\src\datasets" 
df = prepare_dataset(base_dir)
df.to_csv('email_dataset.csv', index=False)
print(f"Dataset prepared with {len(df)} emails")