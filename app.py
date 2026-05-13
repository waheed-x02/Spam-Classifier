import pickle

# Load the trained model
with open('spam_classifier_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the vectorizer
with open('spam_classifier_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def predict_spam(message):
    # Transform the input message using the loaded vectorizer
    message_vector = vectorizer.transform([message])
    
    # Predict using the loaded model
    prediction = model.predict(message_vector)
    
    # Return the result
    if prediction[0] == 1:
        return "The message is SPAM"
    else:
        return "The message is HAM"
    
message = input("Enter a message to classify: ")
result = predict_spam(message)
print(result)
    