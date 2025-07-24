import cv2
from fer import FER

detector = FER()

cap = cv2.VideoCapture(0)

while True:
   
    ret, frame = cap.read()
    
    emotions = detector.detect_emotions(frame)

    for emotion in emotions:
        (x, y, w, h) = emotion["box"]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        dominant_emotion = emotion["emotions"]
        emotion_name = max(dominant_emotion, key=dominant_emotion.get)
        
    
        cv2.putText(frame, emotion_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

  
    cv2.imshow('Emotion Detection', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()