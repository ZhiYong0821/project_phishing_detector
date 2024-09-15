# Project Phishing Detector

## Key information
-Email datasets are obtained from https://spamassassin.apache.org/old/publiccorpus/
--email datasets are labeled as phishing/legitimate (1/0) based off the categorized email datasets downloaded from Spam Assasin Public Corpus
-- Spam Assasin Datasets contains 3 categories of emails:
--  Ham : Legitimate emails (easy to tell)
--  Hard Ham : Legitimate emails (hard to tell)
--  Spam : Phishing/Spam emails 

## Installation requirements
-- 1. Python installed (3.8 or later)
-- 2. virtual environment created and activated
--   2.1 python -m venv env
--   2.2 env\Scripts\activate
-- 3. install required libraries
--   3.1 pip install pandas numpy scikit-learn matplotlib seaborn (see requirements.txt for version used for me (use pip install -r requirements.txt if needed))

## How to use
-data_prep.py file: 
--run this file to generate email_dataset.csv

-data_analysis.py file:
--run this file to generate visualizations using email_dataset.csv (ensure data cleaning is done) file.