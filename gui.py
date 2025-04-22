import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import os
from PIL import Image, ImageTk
import joblib

# ----------------- تحميل النموذج المدرب مسبقاً ------------------
model = joblib.load("model.pkl")  # تأكد أنك خزنت النموذج بهذا الاسم بعد التدريب
le = joblib.load("label_encoder.pkl")

# ----------------- دالة لاستخراج الميزات من الصورة ------------------
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100))  # نفس الحجم اللي درّبت عليه
    return img.flatten()

# ----------------- واجهة المستخدم ------------------
def open_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # عرض الصورة
    img = Image.open(file_path)
    img = img.resize((200, 200))
    img_tk = ImageTk.PhotoImage(img)
    image_label.configure(image=img_tk)
    image_label.image = img_tk

    # التنبؤ
    features = extract_features(file_path).reshape(1, -1)
    prediction = model.predict(features)[0]
    label = le.inverse_transform([prediction])[0]

    result_label.config(text=f"Result: {label.upper()}", fg="green" if label == "real" else "red")

# ----------------- نافذة tkinter ------------------
root = tk.Tk()
root.title("Real and Fake Detection")
root.geometry("500x400")


btn = Button(root, text="Choose Picture", command=open_image, font=("Arial", 14))
btn.pack(pady=10)

image_label = Label(root)
image_label.pack()

result_label = Label(root, text="", font=("Arial", 16))
result_label.pack(pady=10)

root.mainloop()
