import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def main():
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
    X = data['message']
    y = data['is_phishing']

    # Feature extraction using CountVectorizer
    cv = CountVectorizer(stop_words="english")
    X = cv.fit_transform(X)

    # Initialize lists to store accuracy results for each model for all runs
    nb_accuracies = []
    lr_accuracies = []
    dt_accuracies = []
    rf_accuracies = []

    # Run each model 10 times with different random splits
    for run in range(1, 11):
        # Split the data (randomly each time)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=run)

        # Scale the features
        scaler = StandardScaler(with_mean=False) 
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Naive Bayes
        nb_model = GaussianNB()
        nb_model.fit(X_train.toarray(), y_train)  # Naive Bayes expects dense arrays
        nb_predictions = nb_model.predict(X_test.toarray())
        nb_accuracy = accuracy_score(y_test, nb_predictions)
        nb_accuracies.append(nb_accuracy * 100)  # Store in percentage

        # Logistic Regression
        lr_model = LogisticRegression(max_iter=200)
        lr_model.fit(X_train_scaled, y_train)
        lr_predictions = lr_model.predict(X_test_scaled)
        lr_accuracy = accuracy_score(y_test, lr_predictions)
        lr_accuracies.append(lr_accuracy * 100)  # Store in percentage

        # Decision Tree
        dt_model = DecisionTreeClassifier()
        dt_model.fit(X_train, y_train)
        dt_predictions = dt_model.predict(X_test)
        dt_accuracy = accuracy_score(y_test, dt_predictions)
        dt_accuracies.append(dt_accuracy * 100)  # Store in percentage

        # Random Forest
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_predictions)
        rf_accuracies.append(rf_accuracy * 100)  # Store in percentage

        # Print results for this run
        print(f"Run {run}:")
        print(f"  Naive Bayes Accuracy: {nb_accuracy * 100:.2f}%")
        print(f"  Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")
        print(f"  Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")
        print(f"  Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
        print()

    # Prepare to plot the results
    x = np.arange(1, 11)  # X locations for the runs
    width = 0.2  # Width of the bars

    # Create a figure and axis
    plt.figure(figsize=(12, 6))

    # Plotting the bars for each model in each run
    plt.bar(x - 1.5 * width, nb_accuracies, width, label='Naive Bayes', color='blue')
    plt.bar(x - 0.5 * width, lr_accuracies, width, label='Logistic Regression', color='green')
    plt.bar(x + 0.5 * width, dt_accuracies, width, label='Decision Tree', color='orange')
    plt.bar(x + 1.5 * width, rf_accuracies, width, label='Random Forest', color='red')

    # Adding titles and labels
    plt.title('Model Accuracies Over 10 Runs')
    plt.xlabel('Run Number')
    plt.ylabel('Accuracy (%)')
    plt.xticks(x)  # Set x-ticks to run numbers
    plt.ylim(80, 100)  # Adjust the y-axis to focus on higher accuracy ranges
    plt.legend()  # Add a legend to the plot
    plt.grid(axis='y')  # Optional: add grid for better readability

    # Save the plot as a PNG file
    plt.savefig('C:\INF1002\PhishingEmailProject\model_accuracies.png')


    plt.show()  # Display the plot

if __name__ == "__main__":
    main()
