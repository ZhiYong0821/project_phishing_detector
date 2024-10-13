import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from textblob import TextBlob
import nltk

# Ensure that you have NLTK installed and the necessary resources
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('boss_email_dataset_encode_change_after_filter.csv')

# Load unwanted terms from file
with open('../Deploy_+_Visual/UnwantedTerms.txt', 'r') as file:
    unwanted_terms = file.read().splitlines()

# Fill missing values
df['subject'] = df['subject'].fillna('')
df['body'] = df['body'].fillna('')

# Combine subject and body for analysis
df['text'] = df['subject'] + ' ' + df['body']

# Function to clean HTML tags and URLs
def clean_html(text):
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    return text

# Apply HTML cleaning
df['text'] = df['text'].apply(clean_html)

# Initialize the Porter Stemmer for stemming
ps = PorterStemmer()

# Function to perform stemming and remove stop words
def stem_text(text):
    words = text.split()
    stop_words = set(stopwords.words('english'))
    return ' '.join([ps.stem(word) for word in words if word.lower() not in stop_words and word.isalpha()])

# Apply stemming to the text
df['text'] = df['text'].apply(stem_text)

# Data preprocessing
df.drop_duplicates(inplace=True)
df['is_phishing'] = df['is_phishing'].replace({0: 'Not Phishing', 1: 'Phishing'})

# Function to calculate sentiment polarity
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Apply sentiment analysis
df['sentiment'] = df['text'].apply(get_sentiment)

# Plot class distribution with count labels
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

# Generate TF-IDF vectorization and bar plot with unwanted terms excluded
def plot_common_words(data, title, filename):
    stop_words = list(set(unwanted_terms).union(set(stopwords.words('english'))))  # Convert to list
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
def generate_wordcloud(data, title, filename):
    text = ' '.join(data)
    stop_words = set(unwanted_terms).union(set(stopwords.words('english')))  # Combine custom stopwords and NLTK stopwords
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Reds',
                          stopwords=stop_words, max_words=100, contour_width=3, contour_color='black').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, pad=20)
    plt.savefig(filename)
    plt.show()

# Plot Special Character Distribution
def plot_special_char_distribution(data, filename):
    # Recalculate special character count ensuring it's a positive value
    data['special_char_count'] = data['text'].apply(lambda x: sum([1 for char in x if char in '!$%']))
    
    # Ensure we only plot emails that have special characters
    valid_data = data[data['special_char_count'] > 0]
    
    # Debugging: Print size of valid_data
    print(f"Number of emails with special characters: {valid_data.shape[0]}")
    
    # Check if valid_data is empty, if so, skip plotting
    if valid_data.empty:
        print("No special characters found in emails, skipping the plot.")
        return
    
    plt.figure(figsize=(8, 6))
    sns.histplot(data=valid_data, x='special_char_count', hue='is_phishing', kde=False, palette='Set1')
    plt.title('Special Character Usage (Phishing vs Legitimate)')
    plt.xlabel('Special Characters Count')
    plt.ylabel('Frequency')
    plt.legend(title='Email Type', labels=['Legitimate Emails', 'Phishing Emails'])  # Meaningful legend
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Plot Word Count Distribution
def plot_word_count_distribution(data, filename):
    data['word_count'] = data['text'].apply(lambda x: len(x.split()))
    
    plt.figure(figsize=(8, 6))
    sns.histplot(data=data, x='word_count', hue='is_phishing', kde=False, palette='Set2')
    plt.title('Word Count Distribution (Phishing vs Legitimate)')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.legend(title='Email Type', labels=['Legitimate Emails', 'Phishing Emails'])  # Meaningful legend
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Plot Sentiment Distribution
def plot_sentiment_distribution(data, filename):
    plt.figure(figsize=(8, 6))
    sns.histplot(data=data, x='sentiment', hue='is_phishing', kde=True, palette='Set3')
    plt.title('Sentiment Distribution (Phishing vs Legitimate)')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Density')
    plt.legend(title='Email Type', labels=['Legitimate Emails', 'Phishing Emails'])
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Plot Bigrams
def plot_bigrams(data, title, filename):
    stop_words = list(set(unwanted_terms).union(set(stopwords.words('english'))))
    vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=(2, 2), max_features=20)
    
    # Generate bigrams
    bigrams = vectorizer.fit_transform(data)
    bigram_freq_array = bigrams.toarray().sum(axis=0)
    
    # Create a DataFrame for bigrams and their frequencies
    bigram_freq = pd.DataFrame({'bigram': vectorizer.get_feature_names_out(), 'count': bigram_freq_array})
    bigram_freq = bigram_freq.sort_values('count', ascending=False)

    # Check if we have data to plot
    if bigram_freq.empty:
        print("No bigrams found, skipping the plot.")
        return
    
    # Plot the bigrams
    plt.figure(figsize=(12, 6))
    sns.barplot(x='bigram', y='count', data=bigram_freq, palette='Reds_r')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Plot Email Length Distribution
def plot_email_length_distribution(data, filename):
    data['email_length'] = data['text'].apply(len)
    
    plt.figure(figsize=(8, 6))
    sns.histplot(data=data, x='email_length', hue='is_phishing', kde=True, palette='Set1')
    plt.title('Email Length Distribution (Phishing vs Legitimate)')
    plt.xlabel('Email Length (Characters)')
    plt.ylabel('Density')
    plt.legend(title='Email Type', labels=['Legitimate Emails', 'Phishing Emails'])
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Class distribution plot
plot_class_distribution(df, 'class_distribution.png')

# Apply to phishing and legitimate emails
phishing_emails = df[df['is_phishing'] == 'Phishing']['text']
legitimate_emails = df[df['is_phishing'] == 'Not Phishing']['text']

# Plot top 20 common words in phishing and legitimate emails
plot_common_words(phishing_emails, 'Top 20 Words in Phishing Emails', 'phishing_words.png')
plot_common_words(legitimate_emails, 'Top 20 Words in Legitimate Emails', 'legitimate_words.png')

# Create word cloud for phishing emails
generate_wordcloud(phishing_emails, 'Words to Be Aware Of in Phishing Emails', 'phishing_wordcloud.png')

# Create word cloud for legitimate emails
generate_wordcloud(legitimate_emails, 'Words Commonly Used in Legitimate Emails', 'legitimate_wordcloud.png')

# Generate Special Character Distribution plot
plot_special_char_distribution(df, 'special_char_distribution.png')

# Generate Word Count Distribution plot
plot_word_count_distribution(df, 'word_count_distribution.png')

# Generate Sentiment Distribution plot
plot_sentiment_distribution(df, 'sentiment_distribution.png')

# Generate Bigram Plot
plot_bigrams(phishing_emails, 'Top 20 Bigrams in Phishing Emails', 'phishing_bigrams.png')
plot_bigrams(legitimate_emails, 'Top 20 Bigrams in Legitimate Emails', 'legitimate_bigrams.png')

# Generate Email Length Distribution plot
plot_email_length_distribution(df, 'email_length_distribution.png')

print(r'Successful Data Analysis. Visualizations saved under "..Project Phishing Detector\src" .')