from cv2 import VideoCapture
from YOLO_V3_Tiny.model import Predictor
from VLC_Controller.controller import VLCController
from time import sleep

MODEL_PATH = '../model/yolov3-tiny-416-int8.tflite'
VIDEO_PATH = '../sample/showcase.mp4'
PLAYLIST_PATH = '../playlist'

conf = {
    "play": [1],
    "pause": [0],
    "next": [1, 1],
    "previous": [0, 0],
    "stop": [1, 0]
}

cap = VideoCapture(0)
# cap = VideoCapture(VIDEO_PATH)
predictor = Predictor(MODEL_PATH)
controller = VLCController(conf)
controller.setPlaylist(PLAYLIST_PATH)

while(cap.isOpened()):
    ret, frame = cap.read()

    if not ret: break

    detection = predictor.getPredictions(frame)
    print(detection)
    if detection.size > 0:
        controller.configMapper(list(detection))
        sleep(1)