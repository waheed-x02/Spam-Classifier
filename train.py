import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
data = pd.read_csv(
    'spam.csv',
    encoding='latin-1'
)

# Keep only the necessary columns
data = data[['v1', 'v2']]

# Rename columns
data.columns = ['label', 'message']

# Convert Label into binary values
data['label'] = data['label'].map({
    'ham': 0, 
    'spam': 1
})

# Inputs 
x = data['message']

# Outputs
y = data['label']

# Using TfidfVectorizer to convert text into numerical features
tfidf = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2)
)
x = tfidf.fit_transform(x)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

# Using Multinomial Naive Bayes for classification
model = MultinomialNB()
model.fit(x_train, y_train)

# Save model
with open('spam_classifier_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save vectorizer
with open('spam_classifier_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)

# Predicting the labels for the test set
y_pred = model.predict(x_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print the dataset
print(data.head())

# Create a function to classify new messages
def classify_message(message):
    message_transformed = tfidf.transform([message])
    prediction = model.predict(message_transformed)

    message_prob = model.predict_proba(message_transformed)
    print(message_prob)
    
    if prediction[0] == 1:
        return "SPAM"
    else:
        return "HAM"
    
# Test the function with a new message
print(classify_message("Congratulations! You've won a free ticket to the Bahamas. Call now!"))
print(classify_message("WINNER! Claim your free prize now!"))

# Classification Report
cr = classification_report(y_test, y_pred)
print(cr)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
