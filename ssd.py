import cv2
import os
import numpy as np
import tensorflow as tf

model_path = "ssd.pb"
if not os.path.exists(model_path):
    print(f"Model dosyası bulunamadı: {model_path}")
    exit()
try:
    model = tf.ssd.load(model_path)
    infer = model.signatures["serving_default"]
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")
    exit()

labels = ["person","bicycle","car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]  # Örnek etiketler


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Web kamerası açılamadı.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kare alınamadı (video bitimi?). Çıkılıyor...")
        break

    input_tensor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = cv2.resize(input_tensor, (300, 300))
    input_tensor = np.expand_dims(input_tensor, 0)
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint8)

    try:
        output_dict = infer(input_tensor)

        scores = output_dict['detection_scores'].numpy()[0]
        boxes = output_dict['detection_boxes'].numpy()[0]
        classes = output_dict['detection_classes'].numpy()[0].astype(int)
        threshold = 0.5
        for i in range(len(scores)):
            if scores[i] > threshold:
                ymin, xmin, ymax, xmax = boxes[i]
                h, w, _ = frame.shape
                left = int(xmin * w)
                top = int(ymin * h)
                right = int(xmax * w)
                bottom = int(ymax * h)
                
                class_id = classes[i] - 1
                if 0 <= class_id < len(labels):
                    label = labels[class_id]
                else:
                    label = f"Unknown class {class_id}"
                
                confidence = scores[i]
                label_text = f"{label} {confidence:.2f}"

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
    except Exception as e:
        print(f"Çıkarım sırasında hata oluştu: {e}")
        
    cv2.imshow("Webcam", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        print("Escape tuşuna basıldı, çıkılıyor...")
        break

cap.release()
cv2.destroyAllWindows()