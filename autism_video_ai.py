import cv2
import mediapipe as mp
import numpy as np

# Inicializar mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

mp_drawing = mp.solutions.drawing_utils

# Abrir cámara o video
cap = cv2.VideoCapture(0)

eye_contact_counter = 0
frame_counter = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_counter += 1

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS
            )

            # Ejemplo simple de detección de mirada
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]

            if left_eye.x > 0.4 and left_eye.x < 0.6:
                eye_contact_counter += 1

    cv2.putText(
        frame,
        f"Eye Contact Frames: {eye_contact_counter}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Autism AI Screening", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("Total frames:", frame_counter)
print("Eye contact frames:", eye_contact_counter)

ratio = eye_contact_counter / frame_counter

if ratio < 0.2:
    print("Possible social attention risk detected")
else:
    print("Eye contact appears typical")
