from PIL import Image
import face_recognition
import os
from datetime import datetime

"""
pull faces from a directory with images.
after this step you should see and categorize the results.
save the results in .jpg for every face detected with the name of the file and the number of the face detected in the file. 
the saved image is 150x150.
"""

archives_to_pull = os.listdir('to pull faces')
counter = 0
for img in archives_to_pull:
    file_to_pull = img
    # file_to_pull = 'maddy_equipada.jpg'
    image = face_recognition.load_image_file(f'to pull faces/{file_to_pull}')
    face_locations = face_recognition.face_locations(image)

    for face_location in face_locations:
        top, right, bottom, left = face_location

        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image = pil_image.resize((150, 150))
        #this will open all the faces in the default viewer
        # pil_image.show()
        #this will save the face
        counter += 1
        pil_image.save(f"faces pulled/{file_to_pull[:-4]}-{counter} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.jpg")

print(f"pulled {counter} faces")