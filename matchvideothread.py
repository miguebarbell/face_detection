from collections import deque
from multiprocessing.pool import ThreadPool
import face_recognition
import os
import cv2

video = cv2.VideoCapture(0)

known_face_encodings_auto = []
known_face_names_auto = []
for face in os.listdir('faces pulled'):
    known_face_names_auto.append(face.split('-')[0])
    person_file = f'faces pulled/{face}'
    person = face_recognition.load_image_file(person_file)
    person_face_encoding = face_recognition.face_encodings(person)[0]
    known_face_encodings_auto.append(person_face_encoding)


def process_frame(frame):
    test_image = frame
    face_locations = face_recognition.face_locations(test_image)
    face_encodings = face_recognition.face_encodings(test_image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        distances = face_recognition.face_distance(known_face_encodings_auto, face_encoding)

        name = 'Unknown'
        if distances.argmin(axis=0) > 0.6:
            name = known_face_names_auto[distances.argmin(axis=0)]

        #draw box
        top_left = (left, top)
        bottom_right = (right, bottom)
        color = [0, 255, 255]
        frame_thickness = 1
        cv2.rectangle(test_image, top_left, bottom_right, color, frame_thickness)
        #labeling
        top_left = (left, bottom)
        bottom_right = (right, bottom + 22)
        cv2.rectangle(test_image, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(test_image, name.upper(), (left + 10, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return test_image

thread_num = cv2.getNumberOfCPUs()
pool = ThreadPool(processes=thread_num)
pending_task = deque()

while True:
    # Consume the queue.
    while len(pending_task) > 0 and pending_task[0].ready():
    # while len(pending_task) > 0 and pending_task[-1].ready():
        res = pending_task.popleft().get()
        # res = pending_task.pop().get()
        cv2.imshow('threaded video', res)

    # Populate the queue.
    if len(pending_task) < thread_num:
        frame_got, frame = video.read()
        if frame_got:
            # task = pool.apply_async(process_frame, (frame.copy(),))
            task = pool.apply_async(process_frame, (frame,))
            pending_task.append(task)
            # print(pending_task)

    # Show preview.
    if cv2.waitKey(1) == ord('q') or not frame_got:
        break

cv2.destroyAllWindows()