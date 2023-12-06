from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
data = pd.read_csv("your_data.csv")

# Feature engineering
# Assume 'date' is a datetime column
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month

# Assume 'categorical_column' is a categorical feature
data = pd.get_dummies(data, columns=['categorical_column'], drop_first=True)

# Assume 'text_column' is a text feature
# You might use techniques like TF-IDF or word embeddings for text data

# Assume 'target' is the target variable
X = data.drop(['target', 'date'], axis=1)
y = data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build an XGBoost Classifier
model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Display classification report
print(classification_report(y_test, predictions))
