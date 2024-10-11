import os
import pandas as pd
import email
import chardet

def extract_email_features(filepath):
    # Opens the file at given file path in 'read binary' mode, Detects the file encoding
    with open(filepath, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data) # extracts detected encoding from the result
        file_encoding = result['encoding']

    # Open and read email file with detected encoding
    try:
        with open(filepath, 'r', encoding=file_encoding) as file:
            content = file.read()
    except UnicodeDecodeError:
        # If detection fails, try with utf-8
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()

    # Parse email content into a message object (into different parts: body, subject, sender)
    msg = email.message_from_string(content)
    
    # Read the email content and retrieve relevant parts of the email such as body, subject, sender
    subject = msg['subject'] or ''
    sender = msg['from'] or ''
    
    # Handle multipart messages and any potential encoding issues in the body, according to the different types of email content (plain text / multipart message)
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
    # returns a dictionary with all of the relevant parts of content from the emails
    return {
        'subject': subject,
        'from': sender,
        'body': body,
    }

# prepares and organize the collected emails into a structured format
def prepare_dataset(base_directory):
    data = []
    # sort emails according to category of the emails (e.g: phishing/non-phishing email)
    categories = {
        'easy_ham': 0, # legitimate emails (easy to tell)
        'hard_ham': 0, # legitimate emails (harder to tell but still are legit emails)
        'spam': 1 # Confirmed phishing/spam emails
    }

    # starts a loop which run through each category in the dictionary, processing each category of emails at a time.
    for category, is_phishing in categories.items():
        directory = os.path.join(base_directory, category)
        # process and join each file in each email category directory
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            # gets the full path of each file in each category, checks if its actually a file and not a folder
            if os.path.isfile(filepath):
                # extract relevant features from the emails
                features = extract_email_features(filepath)
                features['is_phishing'] = is_phishing # adds the label accordingly
                data.append(features)

    # after processing all emails, returns the list of email (in form of dictionaries) to a pandas dataframe
    return pd.DataFrame(data)

# Changed base_dir code cater to different file paths - YongYing
base_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(base_dir,'datasets')

# assigns the list of email dictionaries to dataframe variable
df = prepare_dataset(data_file_path)

# drops missing values, if any
df.dropna()

# Optionally check again for any remaining missing values
print("Missing values in each column:\n", df.isnull().sum())

# converts prepared dictionary dataset to a CSV file
df.to_csv('boss_email_dataset_encode_change_final.csv', index=False, encoding='utf-8', na_rep='')

print(f"Dataset prepared and saved to current directory.")