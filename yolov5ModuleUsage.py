from yolov5Module import yolov5ObjectDetector
import cv2


yoloModelFile = 'config_files/yolov5n.onnx' # The yolov5 onnx model compatible with opencv
yoloClassConfigFile = "config_files/classes.txt" # The classes names for the detection
image_to_be_detected = "./Inference-Images/test2.jpg" # inference image
detectedImageName = "./Results/detected-image-2.jpg"
imageScaling = 640
verbose = -1


yolov5 = yolov5ObjectDetector(
        model_location=yoloModelFile, 
        class_config_file_location=yoloClassConfigFile, 
        image_to_detect=image_to_be_detected, 
        image_scaling_factor=imageScaling, 
        detected_image_naming=detectedImageName, 
        verbosity=verbose
        )

   
image = yolov5.performDetection()
cv2.imshow("window", image)
cv2.waitKey(0)


