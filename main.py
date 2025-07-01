import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Step 1: Load dataset
df = pd.read_csv('lifestyle_data.csv')

# Step 2: Separate features and target
X = df.drop(columns=['Lifestyle']).values   # all numeric features
y = df['Lifestyle'].values                  # encoded dog breed

# Step 3: Encode the target column (breed name)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 5: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Train the KNN classifier
classifier = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = classifier.predict(X_test)

# Step 8: Evaluate the model
print("\n✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n✅ Accuracy Score:", accuracy_score(y_test, y_pred))