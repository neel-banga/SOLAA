import NLP
import SOLAA.yolo as yolo
import face_det
from audio import say
from audio import listen
import label_reading
import face_det

command = listen()

label = NLP.classify_input(command)

print(label)

# if label == 0:
#     text1 = "Hello Neel this is a test"
#     text.text(text1)

if label == 1:
    yolo.yolo()

elif label == 2:
    face_det.get_person()

# elif label == 3:
#     call.call()

elif label == 4:
    label_reading.read_label()
