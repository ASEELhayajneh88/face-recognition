import cv2
import winsound
import os

# تحميل مصنف الوجه
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# فتح الكاميرا
cap = cv2.VideoCapture(0)

# ضبط دقة الكاميرا (اختياري)
cap.set(3, 640)  # عرض الفيديو
cap.set(4, 480)  # ارتفاع الفيديو

# مجلد لحفظ الصور المكتشفة
output_folder = "detected_faces"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# عداد لعدد الصور التي تم حفظها
saved_face_count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # اكتشاف الوجوه
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # إضافة النص لعدد الوجوه المكتشفة
    face_count = len(faces)
    cv2.putText(frame, f"Faces Detected: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for (x, y, w, h) in faces:
        # رسم مستطيل حول الوجه
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # حفظ الصورة إذا تم اكتشاف وجهض
        face_img = frame[y:y + h, x:x + w]
        face_filename = os.path.join(output_folder, f"face_{saved_face_count + 1}.jpg")
        cv2.imwrite(face_filename, face_img)
        saved_face_count += 1

        # إضافة تأثير صوتي عند اكتشاف الوجه
        winsound.Beep(1000, 200)  # تردد 1000 هرتز لمدة 200 مللي ثانية

    # عرض الصورة بالكاميرا
    cv2.imshow('Face Detection', frame)

    # إذا ضغطت "q" ستغلق الكاميرا
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# إغلاق الكاميرا والنوافذ المفتوحة
cap.release()
cv2.destroyAllWindows()
