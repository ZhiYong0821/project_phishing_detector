import os
import pandas as pd
import email

# extracts features from email files for analysis
def extract_email_features(filepath):
    with open(filepath, 'r', encoding='latin-1') as file: # open and read email file
        content = file.read()
    #parses email content
    msg = email.message_from_string(content)
    # read the email content and retrieve relevant parts of the email such as subject, sending and body
    return {
        'subject': msg['subject'] or '',    # get email sender or empty string if none
        'from': msg['from'] or '',          # get email sender or empty string if none
        'body': msg.get_payload() if msg.is_multipart() else msg.get_payload(), # get email body content
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

#base_dir = r"C:\Users\Admin\Documents\Code\School Assignments\Project Phishing Detector\src\datasets" 
base_dir = '/Users/zhiyong/project_phishing_detector/src/datasets'

#assigns the list of email dictionaries to df variable
df = prepare_dataset(base_dir)

# converts prepared dictionary dataset to a CSV file
df.to_csv('email_dataset.csv', index=False)

print(f"Dataset prepared with {len(df)} emails")