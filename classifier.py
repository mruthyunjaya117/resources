import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle
import seaborn as sns

# added for comment

# Step 1: Load and preprocess your dataset
df = pd.read_excel('ims_training_data.xlsx')  

# Step 2: Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['detailed_decription'])  
y = df['assigned_support_group']  # Replace 'label_column' with the actual column containing your labels

# Step 3: Handling Class Imbalance (Applying Class Weights)
class_weights = dict(df['assigned_support_group'].value_counts(normalize=True))

# Step 4: Training the SVM Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_classifier = SVC(kernel='linear', class_weight=class_weights)  # Adding class weights
svm_classifier.fit(X_train, y_train)


# Step 5: Evaluation and Performance Metrics
y_pred = svm_classifier.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)

filename = 'svm_model.pkl'  # Specify the filename and path for saving the model
with open(filename, 'wb') as file:
    pickle.dump(svm_classifier, file)