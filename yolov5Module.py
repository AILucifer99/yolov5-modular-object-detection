import numpy as np
import cv2


class yolov5ObjectDetector :
    def __init__(self, model_location, class_config_file_location, 
                    image_to_detect, image_scaling_factor, detected_image_naming, verbosity) :
        super(yolov5ObjectDetector, self).__init__()
        self.model_location = model_location
        self.class_config_file_location = class_config_file_location
        self.image_to_detect = image_to_detect
        self.image_scaling_factor = image_scaling_factor
        self.detected_image_naming = detected_image_naming
        self.verbosity = verbosity


    def performDetection(self) :
        net = cv2.dnn.readNet(self.model_location)

        # step 2 - feed a 640x640 image to get predictions

        def format_yolov5(frame):

            row, col, _ = frame.shape
            _max = max(col, row)
            result = np.zeros((_max, _max, 3), np.uint8)
            result[0:row, 0:col] = frame
            return result

        image = cv2.imread(self.image_to_detect)

        input_image = format_yolov5(image) # making the image square
        blob = cv2.dnn.blobFromImage(input_image , 1/255.0, (640, 640), swapRB=True)
        net.setInput(blob)
        predictions = net.forward()

        # step 3 - unwrap the predictions to get the object detections 

        class_ids = []
        confidences = []
        boxes = []

        output_data = predictions[0]

        image_width, image_height, _ = input_image.shape
        x_factor = image_width / self.image_scaling_factor
        y_factor =  image_height / self.image_scaling_factor

        for r in range(25200):
            row = output_data[r]
            confidence = row[4]
            if confidence >= 0.4:

                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if (classes_scores[class_id] > .25):

                    confidences.append(confidence)

                    class_ids.append(class_id)

                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        class_list = []

        with open(self.class_config_file_location, "r") as f:
            class_list = [cname.strip() for cname in f.readlines()]


        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

        result_class_ids = []
        result_confidences = []
        result_boxes = []

        for i in indexes:
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])

        for i in range(len(result_class_ids)):

            box = result_boxes[i]
            class_id = result_class_ids[i]

            cv2.rectangle(image, box, (0, 255, 255), 2)
            cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 255), -1)
            cv2.putText(image, class_list[class_id], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))

        cv2.imwrite(self.detected_image_naming, image)

        if self.verbosity > -1 :
            cv2.imshow("output", image)

        return image  


# For Local Usage    
# if __name__ == '__main__' :

#     yoloModelFile = 'config_files/yolov5s.onnx' # The yolov5 onnx model compatible with opencv
#     yoloClassConfigFile = "config_files/classes.txt" # The classes names for the detection
#     image_to_be_detected = "./test1.jpg" # inference image
#     detectedImageName = "./inference.jpg"
#     imageScaling = 640
#     verbose = -1

#     yolov5 = yolov5ObjectDetector(
#         model_location=yoloModelFile, 
#         class_config_file_location=yoloClassConfigFile, 
#         image_to_detect=image_to_be_detected, 
#         image_scaling_factor=imageScaling, 
#         detected_image_naming=detectedImageName, 
#         verbosity=verbose
#         )
    
#     image = yolov5.performDetection()
#     cv2.imshow("window", image)
#     cv2.waitKey(0)

