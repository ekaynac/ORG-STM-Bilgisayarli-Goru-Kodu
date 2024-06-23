from TFmodel import *
from PIL import Image

DETECTION_THRESHOLD = 0.6
TFLITE_MODEL_PATH = r"D:\GitHub\Meturone\Meturone\UORG\Object Detection\Model\efficientdet_lite0-portablegenerator.tflite"

# Load the TFLite model
options = ObjectDetectorOptions(
      num_threads=13,
      score_threshold=DETECTION_THRESHOLD,
      max_results=1
)
detector = ObjectDetector(model_path=TFLITE_MODEL_PATH, options=options)


if __name__=="__main__":
    cap = cv2.VideoCapture(r"D:\GitHub\Meturone\Meturone\UORG\Object Detection\Model\output.avi")
    while(True):
        ret, frame = cap.read()
        if ret:
            image_np = np.asarray(frame)
            image_np = cv2.resize(image_np,(512,512))
            # Run object detection estimation using the model.
            detections = detector.detect(image_np)

            # Draw keypoints and edges on input image
            image_np = visualize(image_np, detections)

            # Show the detection result
            img = Image.fromarray(image_np)
            
            cv2.imshow("detection",image_np)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        key = cv2.waitKey(1)
        if key==27:
            break
    cap.release()
    cv2.destroyAllWindows()
