
import cv2
import numpy as np
import os
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
def load_images(folder, label):
    images, labels = [], []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (100, 100))
            images.append(img.flatten())
            labels.append(label)
    return images, labels

real_images, real_labels = load_images("dataset/real", "real")
fake_images, fake_labels = load_images("dataset/fake", "fake")
X = np.array(real_images + fake_images)
y = np.array(real_labels + fake_labels)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train SVM model
model = SVC(kernel='linear', C=1.0, probability=True)
model.fit(X_train, y_train)
#تقييم النموذج
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))


# Cross-validation
scores = cross_val_score(model, X, y_encoded, cv=5)
print("Cross-validation scores:", scores)
print("Average CV Score:", scores.mean())

# Save model and encoder
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
# 1. تقرير التصنيف الكامل
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

