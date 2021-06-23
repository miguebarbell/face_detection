import face_recognition
from PIL import Image, ImageDraw
from datetime import datetime
import os


"""
return an image with the names of the images recognized and compared with know pulled faces.
will use all the faces in the faces pulled directory.
create a big database with different faces, shapes, hairstyle of the same persons for a better identification.
could be an error from reading a face, that doesnt process it.
"""


#load database of people
# person_1_file = 'faces pulled/Madeline Q-wedding4.jpg'
# person_1 = face_recognition.load_image_file(person_1_file)
# person_1_face_encoding = face_recognition.face_encodings(person_1)[0]
#
# person_2_file = 'faces pulled/Peter Q-wedding1.jpg'
# person_2 = face_recognition.load_image_file(person_2_file)
# person_2_face_encoding = face_recognition.face_encodings(person_2)[0]
#
# person_3_file = 'faces pulled/Felipe A-wedding1.jpg'
# person_3 = face_recognition.load_image_file(person_3_file)
# person_3_face_encoding = face_recognition.face_encodings(person_3)[0]
#
#
# person_4_file = 'faces pulled/Felipe A-wedding2.jpg'
# person_4 = face_recognition.load_image_file(person_4_file)
# person_4_face_encoding = face_recognition.face_encodings(person_4)[0]
#
#
# #  Create arrays of encodings and names
# known_face_encodings = [
#     person_1_face_encoding,
#     person_2_face_encoding,
#     person_3_face_encoding,
#     person_4_face_encoding
#
# ]
#
# known_face_names = [
#     "Madeline Q",
#     "Peter Q",
#     "Felipe A",
#     "Felipe A"
# ]

#automate the faces.
known_face_encodings_auto = []
known_face_names_auto = []
for face in os.listdir('faces pulled'):
    known_face_names_auto.append(face.split('-')[0])
    person_file = f'faces pulled/{face}'
    # print(f'person_file: {person_file}')
    person = face_recognition.load_image_file(person_file)
    # print(f'person: {person}')
    person_face_encoding = face_recognition.face_encodings(person)[0]
    known_face_encodings_auto.append(person_face_encoding)
    # known_face_encodings_auto.append(face_recognition.face_encodings(face_recognition.load_image_file(f'faces pulled/{face}')[0]))
    # known_face_encodings_auto.append(face_recognition.face_encodings(face_recognition.load_image_file(f'faces pulled/{face}'))[0])

# print(known_face_encodings_auto)

# Load test image to find faces in
# test_image_file = 'to pull faces/Maddy_migue_048.jpg'

unworked_images = [testimage for testimage in os.listdir('unworked images')]

for unworked_image in unworked_images:
    # find face in unworked image
    test_image_file = f'unworked images/{unworked_image}'
    test_image = face_recognition.load_image_file(test_image_file)
    # print(type(test_image))
    face_locations = face_recognition.face_locations(test_image)
    # face_locations = face_recognition.face_locations(test_image, model='cnn')
    # face_locations = face_recognition.face_locations(test_image, model='hog')

    # print(face_locations)
    face_encodings = face_recognition.face_encodings(test_image, face_locations)
    # print(face_encodings)
    # conver to PIL
    pil_image = Image.fromarray(test_image)
    # create a image draw instance
    draw = ImageDraw.Draw(pil_image)

    # find matches from faces and unworked image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # print(face_encoding)
        # print(type(face_encoding))

        # matches = face_recognition.compare_faces(known_face_encodings_auto, face_encoding)
        distances = face_recognition.face_distance(known_face_encodings_auto, face_encoding)
        # print(f'MATCHES= {matches}')
        # print(f'DISTANCES= {distances}')
        # print(f'min value= {distances.argmin(axis=0)}')

        name = 'Unknown'
        # if True in matches:
        if distances.argmin(axis=0) > 0.6:
            # match_index = matches.index(True)
            # name = known_face_names_auto[match_index]
            name = known_face_names_auto[distances.argmin(axis=0)]

        #draw box
        draw.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0))
        #draw label
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(255, 255, 0), outline=(255, 255, 0))
        #insert the name
        draw.text((left + 6, bottom - text_height - 5), name, fill=(0, 0, 0))
    del draw
    pil_image.save(f"worked images/{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.jpg")

# test_image = face_recognition.load_image_file(test_image_file)
#
# # Find faces in test image
# face_locations = face_recognition.face_locations(test_image)
# face_encodings = face_recognition.face_encodings(test_image, face_locations)
#
# # Convert to PIL format
# pil_image = Image.fromarray(test_image)
#
# # Create a ImageDraw instance
# draw = ImageDraw.Draw(pil_image)
#
# # Loop through faces in test image
# for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#     matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#
#     name = "Unknown Person"
#
#     # If match
#     if True in matches:
#         first_match_index = matches.index(True)
#         name = known_face_names[first_match_index]
#
#     # Draw box
#     draw.rectangle(((left, top), (right, bottom)), outline=(255,255,0))
#
#     # Draw label
#     text_width, text_height = draw.textsize(name)
#     draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,0))
#     draw.text((left + 6, bottom - text_height - 5), name, fill=(0,0,0))
#
# del draw
#
# # Display image
# pil_image.show()
#
# # Save image
# pil_image.save(f"worked images/{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.jpg")
