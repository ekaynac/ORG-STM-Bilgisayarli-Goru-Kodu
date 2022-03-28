#@title Run object detection and show the detection results
from modelclass import *
from PIL import Image
import numpy as np
import os
import time

DETECTION_THRESHOLD = 0.25
NUM_THREADS = 8

PATH = "/home/Github/ORG-STM-Bilgisayarli-Goru-Kodu/2" # 2.py belgesinin adresi / diğer adreslerde buraya bağlı olarak tanımlanmıştır
TFLITE_MODEL_PATH = PATH + "/efficientdet_lite0-portablegenerator-augmented.tflite" #.tflite dosyasının adresi
DATA_PATH = PATH + "/bos" #resimlerin bulunduğu klasörün adresi
OUT_PATH = PATH + "/out.txt" #kordinatların .txt formatındaki çıktısı / gerekli dosyayı kendisi yaratır

list_image = os.listdir(DATA_PATH)
list_detections = []
# Load the TFLite model
options = ObjectDetectorOptions(
      num_threads=NUM_THREADS,
      score_threshold=DETECTION_THRESHOLD
      
)
detector = ObjectDetector(model_path=TFLITE_MODEL_PATH, options=options)
tick = time.time()
j=0
for i in list_image:
    if j == 0:
        print("Detection Started!")
    j+=1
    img_path = os.path.join(DATA_PATH,i)
    img = cv2.imread(img_path)
    image_np = np.asarray(img)
    #image_np = cv2.resize(img,(512,512))
    # Run object detection estimation using the model.
    detections = detector.detect(image_np)
    # Draw keypoints and edges on input image
    #image_np = visualize(image_np, detections)
    try:
        detections = detections[0][0]
        w = detections[2] - detections[0]
        h = detections[3] - detections[1]
        list_detections.append("{} {} {} {} {}".format(i[:-4],detections[0],detections[1],w,h))
    except:
        list_detections.append("{}".format(i[:-4]))
        pass
    # Show the detection result
    #img = Image.fromarray(image_np)
            
    #cv2.imshow("detection",image_np)
    count = len(list_image) - j
    if count == 0:
        tock= time.time()
        print("{} images processed in {} seconds!".format(j,tock-tick))
    elif count % 10 == 0:
        print("Remaining image count: {}".format(count))

    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()

with open(OUT_PATH, "w") as output:
    for line in list_detections:
        output.write(line+"\n")