from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_objects(image):
    results = model(image)

    objects = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            name = model.names[class_id]

            objects.append({
                "object": name,
                "confidence": round(confidence, 2)
            })

    return objects