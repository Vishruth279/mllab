
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
# Load the dataset
# Assuming the data is in a CSV file with columns: feature1, feature2, feature3, label
# Replace 'your_dataset.csv' with the actual filename
data = pd.read_csv('/home/root1/data_vowel_bayes.csv', header=None, names=['feature1', 'feature2', 'feature3', 'label'])
# Split the data into features (X) and labels Yes
X = data[['feature1', 'feature2', 'feature3']]
y = data['label']
# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the Naive Bayes classifier
classifier = GaussianNB()
# Train the classifier on the training data
classifier.fit(X_train, y_train)
# Make predictions on the testing data
y_pred = classifier.predict(X_test)
# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
# Calculate precision, recall, and F1 score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1_score = metrics.f1_score(y_test, y_pred, average='weighted')
# Print accuracy, precision, recall, and F1 score
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1_score}')
# Calculate ROC curve and AUC
y_probs = classifier.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs, pos_label=1)
roc_auc = auc(fpr, tpr)
# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
