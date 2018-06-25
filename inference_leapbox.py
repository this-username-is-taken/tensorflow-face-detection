#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

THRESHOLD = 0.5

import sys
import time
import numpy as np
import tensorflow as tf
import cv2
import socket
import traceback


from utils import label_map_util
from utils import visualization_utils_color as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

LOG_PATH = "/home/nvidia/Projects/tensorflow-face-detection/python_log"

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def write_to_file(filename, s, end = "\n"):
    file = open(filename, 'a')
    file.write(now() + ": " + str(s) + end)
    file.close()

def now():
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())

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

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
        print('inference time cost: {}'.format(elapsed_time))

        return (boxes, scores, classes, num_detections)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        camID = 0
    else:
        camID = sys.argv[1]

    write_to_file(LOG_PATH, "Start")

    try:
    	tDetector = TensoflowFaceDector(PATH_TO_CKPT)

    	write_to_file(LOG_PATH, "Tensorflow ready")
    	cap = cv2.VideoCapture(camID)
    	windowNotSet = True

    	write_to_file(LOG_PATH, "Cam ready")
    except:
        write_to_file(LOG_PATH, "Exception: " + traceback.format_exc())

    serversocket = None

    while True:
        try:
            ##########################
            if serversocket is None:
                print("Preparing port...")
                port = 12345
                serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                while port < 20000:
                    try:
                        serversocket.bind(('192.168.1.6', port))
                        break
                    except:
                        port = port + 1

                if port > 19990:
                    print("ERROR: could not find port")
                    exit(-1)
                print("Waiting on port " + str(port))
                write_to_file(LOG_PATH, port)

                serversocket.listen(1)
                connection, address = serversocket.accept()
                print("Connected with " + str(address))
                write_to_file(LOG_PATH, "Connected with " + str(address))
            ##########################

            ret, image = cap.read()
            if ret == 0:
                break

            [h, w] = image.shape[:2]
            image = cv2.flip(image, 1)

            (boxes, scores, classes, num_detections) = tDetector.run(image)

            faces = []
            for i in range(num_detections):
                if scores[0][i] > THRESHOLD:
                    faces.append(boxes[0][i])

            ##########################
            buf = ""
            for face in faces:
                buf += " ".join(map(lambda x: str(x), face)) + "|"
            print(buf)
            connection.send(buf)
            write_to_file(LOG_PATH, buf)
            ##########################
        except:
            write_to_file(LOG_PATH, "Exception: " + traceback.format_exc())
            serversocket = None


    cap.release()
    serversocket.close()

