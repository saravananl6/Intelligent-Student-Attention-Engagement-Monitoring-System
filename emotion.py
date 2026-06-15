import cv2
from deepface import DeepFace

# Map DeepFace emotions → Your project emotions
def map_emotion(emotion):
    if emotion in ["fear", "surprise"]:
        return "CONFUSED 😕"
    elif emotion in ["angry", "disgust"]:
        return "FRUSTRATED 😤"
    elif emotion in ["neutral"]:
        return "BORED 😐"
    elif emotion in ["happy"]:
        return "ENGAGED 😊"
    elif emotion in ["sad"]:
        return "LOW ENERGY 😞"
    else:
        return "UNKNOWN"

# Start webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze emotion
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )

        emotion = result[0]['dominant_emotion']
        mapped_emotion = map_emotion(emotion)

        # Display text
        cv2.putText(
            frame,
            f"Emotion: {mapped_emotion}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    except Exception as e:
        print("Error:", e)

    # Show webcam
    cv2.imshow("LearnFlow Emotion Detection", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()