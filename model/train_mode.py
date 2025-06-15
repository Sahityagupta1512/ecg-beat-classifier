import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Load data
normal_beats = np.load(r"/workspaces/ecg-beat-classifier/Data/normal_beats.npy")
abnormal_beats = np.load(r"/workspaces/ecg-beat-classifier/Data/abnormal_beats.npy")

# Labels: 0 = Normal, 1 = Abnormal
X = np.concatenate([normal_beats, abnormal_beats])
y = np.concatenate([[0]*len(normal_beats), [1]*len(abnormal_beats)])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=["Normal", "Abnormal"]))

joblib.dump(clf, r"/workspaces/ecg-beat-classifier/model/ecg_classifier.pkl")
print("💾 Model saved to model/ecg_classifier.pkl")
