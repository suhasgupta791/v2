#!/usr/bin/python3
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import numpy as np
import tensorflow as tf
import cv2 as cv
import paho.mqtt.client as mqtt

from utils import label_map_util
from utils import visualization_utils_color as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/data/model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/data/protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

class TensoflowFaceDector(object):
    def __init__(self, PATH_TO_CKPT):
        """Tensorflow detector
        """

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True

    def run(self, image):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """
        image_np = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time
        print('Faces detected: {} | inference time cost: {}'.format(len(np.where(scores>0.5)[0]),elapsed_time))

        return (boxes, scores, classes, num_detections)

    def crop_face(self,box,img):
        """box: coordinates of bounding box containing the detected face in the image
           img: image 
           croppedFrame: cropped image
        """
        y1,x1,y2,x2 = box.astype(int)  # note the order of x and y coordinates !!
        croppedFrame = img[y1:y2,x1:x2]
        size = min(img.shape[0],img.shape[1])
        croppedFrame=cv.resize(croppedFrame,(size,size))
        return croppedFrame

    def displayImage(self,windowNotSet,image):
        w,h = image.shape[0],image.shape[1]
        if windowNotSet is True:
           cv.namedWindow("Cropped face (%d, %d)" % (w, h), cv.WINDOW_NORMAL)
           windowNotSet = False
        cv.imshow("Cropped face (%d, %d)" % (w,h), image)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print ("usage:%s (cameraID | filename) Detect faces\
 in the video example:%s 0"%(sys.argv[0], sys.argv[0]))
        exit(1)

    try:
    	camID = int(sys.argv[1])
    except:
    	camID = sys.argv[1]
    
    tDetector = TensoflowFaceDector(PATH_TO_CKPT)
    DETECTION_THRESHOLD = 0.5 # set a threshold for displaying detected faces

    # Setup Mqtt client to host communication
    host_ip='local_mqtt_broker' # docker ip address of the broker
    port=1883 # mqtt port
    keepalive=120 # timeout
    topic="face_detection"
    client=mqtt.Client()
    client.connect(host_ip,port,keepalive)

    cap = cv.VideoCapture(camID)

    windowNotSet = True
    while True: # read frames from video indefinitely 
        ret, frame = cap.read()
        if ret == 0:
            break

        # Flip to maintain orientatio`n 
        frame = cv.flip(frame, 1)

        # Run the detector on a frame
        (boxes, scores, classes, num_detections) = tDetector.run(frame)

        detected_indeces = np.where(scores[0] > DETECTION_THRESHOLD)[0]  # only update frame when  detection score is above threshold
        for box in boxes[0][detected_indeces]:
            normalized_box = box * np.array([frame.shape[0], frame.shape[1], frame.shape[0], frame.shape[1]])
            cropped_face = tDetector.crop_face(normalized_box,frame)
            tDetector.displayImage(windowNotSet,cropped_face)
            msg = cropped_face.tobytes()  # convert to byte stream
            client.publish(topic,msg)

        k = cv.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break
    cap.release()
