# collect the data
import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):  # if the directory doesn't exist, make it
    os.makedirs(DATA_DIR)

NumOfClasses = 5  # the number of signs to collect
sampleSize = 500  # number of samples per type

cap = cv2.VideoCapture(0)

for j in range(NumOfClasses):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Press "Q" to start recording! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < sampleSize:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
