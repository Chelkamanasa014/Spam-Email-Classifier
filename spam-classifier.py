from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Training data
emails = [
    "Win a free prize now",
    "Congratulations you won money",
    "Claim your free gift",
    "Click here to win cash",
    "Get a free lottery ticket",
    "Meeting is scheduled for tomorrow",
    "Please submit your assignment",
    "Can we meet at college",
    "Send me the project report",
    "Your class starts at nine"
]

labels = [
    "Spam", "Spam", "Spam", "Spam", "Spam",
    "Not Spam", "Not Spam", "Not Spam",
    "Not Spam", "Not Spam"
]

# Create the machine learning model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train the model
model.fit(emails, labels)

print("================================")
print("      SPAM EMAIL CLASSIFIER")
print("================================")
print("Model trained successfully!")

# Predict sample emails
test_emails = [
    "You won free money",
    "Please send the assignment"
]

print("\nSample Predictions:")
for email in test_emails:
    result = model.predict([email])[0]
    print("Email:", email)
    print("Result:", result)
    print()

# Test your own message
message = input("Enter an email message: ")
result = model.predict([message])[0]

print("Classification:", result)
print("================================")