import cv2
import mediapipe as mp
import pickle
import numpy as np

modelDict = pickle.load(open('./model.p', 'rb'))  # loading the model
model = modelDict['model']


frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'Y', 4: 'L'}

cv2.resizeWindow('frame', 200, 600)  # Set the desired width and height

while True:
    data_aux = []
    xDup = []
    yDup = []
    ret, frame = cap.read()
    H, W, _ = frame.shape  # _ to ignore the number of color channels

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)  # detecting all landmarks in this image
    if results.multi_hand_landmarks:  # all landmarks
        # to show the landmarks
        # for hand_landmarks in results.multi_hand_landmarks:
        #     mp_drawing.draw_landmarks(
        #         frame,  # image to draw
        #         hand_landmarks,  # model output
        #         mp_hands.HAND_CONNECTIONS,  # hand connections
        #         mp_drawing_styles.get_default_hand_landmarks_style(),
        #         mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                xDup.append(x)
                yDup.append(y)

        # setting dimensions for the square
        x1 = int(min(xDup) * W) - 10
        y1 = int(min(yDup) * H) - 10

        x2 = int(max(xDup) * W) - 10
        y2 = int(max(yDup) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predictedChar = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)  # dimensions, color, width
        cv2.putText(frame, predictedChar, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
