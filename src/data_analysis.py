import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

# load the prepared email data from earlier step
df = pd.read_csv('cleaned_email_dataset.csv')

# if there are any missing values in 'subject' and 'body' columns, fill it with empty strings
df['subject'] = df['subject'].fillna('')
df['body'] = df['body'].fillna('')

# combines/join content in subject and body for text analysis 
df['text'] = df['subject'] + ' ' + df['body']


plt.figure(figsize=(8, 6)) # sets up new figure with size of 8 by 6 inches
sns.countplot(x='is_phishing', data=df, hue='is_phishing', palette='dark', legend=False) # sets up plot with relevant x and y axis with color pallet preferences
plt.title('Distribution of Phishing vs Legitimate Emails')
plt.xlabel('Is Phishing')
plt.savefig('class_distribution.png')
plt.close()

# function for plotting common words found in the prepared legitimate/phishing email datasets in the form of a barplot visualization
def plot_common_words(data, title, filename):
    # defines a list of custom stop words that are common in email headers/HTML formatting, not useful for analysis of whether the emails are phishing/legitimate
    custom_stop_words = ['http', 'com', 'width', 'font', 'height', 'www', 'https']

    # combines default english stop words list with prior custom stop words list for a more refined visualization of common words associated with phishing/legitimate emails
    stop_words = CountVectorizer(stop_words='english').get_stop_words()
    stop_words = list(stop_words) + custom_stop_words

    # initialzies the 'CountVectorizer' object to transform the text data of the dataset into word counts excluded the stop words, limit output to top 20 most frequent words
    vectorizer = CountVectorizer(stop_words=stop_words, max_features=20)
    word_count = vectorizer.fit_transform(data)
    word_freq_array = word_count.toarray().sum(axis=0)
    
    # creates a pandas DataFrame for each row to correspond to a word, count column shows the amount of times the particular word appears in the emails
    word_freq = pd.DataFrame({'count': word_freq_array}, index=vectorizer.get_feature_names_out())
    word_freq = word_freq.sort_values('count', ascending=False)

    # word frequency plotting details
    plt.figure(figsize=(12, 6))
    sns.barplot(x=word_freq.index, y='count', data=word_freq, hue=word_freq.index, palette='dark', legend=False)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# filters all phishing labeled email content text and store it in the phishing_email variable
phishing_emails = df[df['is_phishing'] == 1]['text']
plot_common_words(phishing_emails, 'Top 20 Most Common Words in Phishing Emails', 'phishing_common_words.png')

# filters all legitimate labeled email content text and store it in the legitimate_emails variable
legitimate_emails = df[df['is_phishing'] == 0]['text']
plot_common_words(legitimate_emails, 'Top 20 Most Common Words in Legitimate Emails', 'legitimate_common_words.png')

print(r'Successful Data Analysis. Visualizations saved under "..Project Phishing Detector\src" .')