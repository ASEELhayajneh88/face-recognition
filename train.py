import os
import cv2
import pickle
import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# إعدادات
image_size = (100, 100)  # حجم الصور بعد التغيير
data_dir = "dataset"  # مجلد البيانات
classes = ['real', 'fake']  # الفئات التي نصنفها

X = []
y = []

# تحميل البيانات
for label in classes:
    folder_path = os.path.join(data_dir, label)
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, image_size)
            X.append(img.flatten())
            y.append(label)

X = np.array(X)
y = np.array(y)

# ترميز الفئات
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# تدريب النموذج
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# تقييم النموذج
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# حفظ النموذج
joblib.dump(model, "model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ تم حفظ النموذج والـ Label Encoder بنجاح.")

# ========================== إضافة التنبؤ بصورة جديدة ==========================

# تحميل النموذج المحفوظ
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# افترض أن لديك صورة جديدة تحتاج إلى تصنيفها
img_path = "new_image.jpg"  # ضع مسار الصورة الجديدة هنا
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is not None:
    img = cv2.resize(img, image_size)
    img_flattened = img.flatten().reshape(1, -1)  # تحويل الصورة إلى شكل متوافق مع النموذج

    # التنبؤ باستخدام النموذج المحمّل
    prediction = model.predict(img_flattened)
    predicted_label = label_encoder.inverse_transform(prediction)
    print(f"تم تصنيف الصورة إلى الفئة: {predicted_label[0]}")
else:
    print("لم يتم العثور على الصورة.")
