import cv2
import cvzone
import math
from ultralytics import YOLO
import function
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

model = YOLO("playingCards.pt")
classNames = [
    '10C', '10D', '10H', '10S',
    '2C', '2D', '2H', '2S',
    '3C', '3D', '3H', '3S',
    '4C', '4D', '4H', '4S',
    '5C', '5D', '5H', '5S',
    '6C', '6D', '6H', '6S',
    '7C', '7D', '7H', '7S',
    '8C', '8D', '8H', '8S',
    '9C', '9D', '9H', '9S',
    'AC', 'AD', 'AH', 'AS',
    'JC', 'JD', 'JH', 'JS',
    'KC', 'KD', 'KH', 'KS',
    'QC', 'QD', 'QH', 'QS',
]
hand = []
while True:
    sucess, img = cap.read()
    results = model(img,stream = True)
    detected_hand = []
    predictions = model(img,stream=True)
    for r in predictions:
        for box in r.boxes:

            x1,y1,x2,y2 = map(int,box.xyxy[0])
            cls = int(box.cls[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            card = model.names[cls]
            cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1))
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0,x1), max(35,y1)), scale = 1, thickness = 1)
            if conf >= 0.5 and card not in detected_hand:
                detected_hand.append(card)
    if len(detected_hand) == 5:
        results = function.findPokerHand(detected_hand)
        cvzone.putTextRect(
            img,
            results,
            (30,60),
            scale=2,
            thickness=2
        )
    cv2.imshow("Image", img)
    cv2.waitKey(1)