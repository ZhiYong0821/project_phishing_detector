# Project Phishing Detector

This project is focused on detecting phishing emails using data analytics and machine learning. By analyzing patterns and features commonly associated with phishing emails, this project aims to build a model capable of distinguishing between legitimate and phishing emails.

## Technology Stack

- Python
- Pandas
- Scikit-Learn
- MatPlotLib

## Key information

- Email datasets are obtained from https://spamassassin.apache.org/old/publiccorpus/
- Email datasets are labeled as phishing/legitimate (1/0) based off the categorized email datasets downloaded from Spam Assasin Public Corpus
- Spam Assasin Datasets contains 3 categories of emails:
  - Ham : Legitimate emails (easy to tell)
  - Hard Ham : Legitimate emails (hard to tell)
  - Spam : Phishing/Spam emails

## Installation requirements

- Python installed (3.8 or later)
- Create virtual environment and activate it

```
python -m venv venv
env\Scripts\activate
```

- Install required libraries

```
pip install -r requirements.txt
```

## How to use

- Run the data_prep.py file to generate email_dataset.csv
- Run the data_analysis.py file to generate visualizations using email_dataset.csv (ensure data cleaning is done first before running this file).
