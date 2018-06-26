#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import math
import numpy as np
import tensorflow as tf
import cv2

if len(sys.argv) != 2:
  print("Need video path")
  exit(1)
video_path = sys.argv[1]

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils_color as vis_util


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

jump_window = []
last_jumped_frame = 0
last_face = (0, 0, 0, 0)
JUMP_WINDOW_SIZE = 10
JUMP_THRESHOLD = 0.05
JUMP_INTERVAL = 10

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

cap = cv2.VideoCapture(video_path)
out = None

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(graph=detection_graph, config=config) as sess:
    frame_num = 1490
    frame_count = 0
    out_file = open(video_path + ".txt", 'w')

    while frame_num:
      frame_count += 1
      frame_num -= 1
      ret, image = cap.read()
      if ret == 0:
          break

      if out is None:
          [h, w] = image.shape[:2]
          out = cv2.VideoWriter(video_path + ".out.avi", 0, 25.0, (w, h))


      image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')

      # Actual detection.
      start_time = time.time()
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      elapsed_time = time.time() - start_time
      #print('inference time cost: {}'.format(elapsed_time))



      faces = []
      face_scores = []
      face_classes = []
      
      face_closest_y = 0
      face_dist = 10000

      for i in range(num_detections):
          box = boxes[0][i]
          score = scores[0][i]
          if score > 0.5:
              dist = 0
              for j in range(4):
                  dist += abs(box[j] - last_face[j])
              print(dist)
              if box[1] > 0.4 and box[3] < 0.6 and dist < face_dist:
                  faces.append(box)
                  face_scores.append(score)
                  face_classes.append(1)
                  face_closest_y = box[0]

                  face_dist = dist
                  last_face = box

      jump_window.append(face_closest_y)

      ##########################
      buf = " "
      for face in faces:
          buf += " ".join(map(lambda x: str(x), face)) + "|"
      print(buf)
      out_file.write(buf + "\n")

      print(jump_window)

      size = len(jump_window)
      if (size > JUMP_WINDOW_SIZE):
        jump_window = jump_window[size - 10:size]
      cur_max = -1
      cur_min = 2
      idx_max = -1
      idx_min = -1
      idx = 0
      for jump in jump_window:
        if jump > cur_max:
          cur_max = jump
          idx_max = idx
        if jump < cur_min:
          cur_min = jump
          idx_min = idx
        idx += 1

      if cur_max - cur_min > JUMP_THRESHOLD and idx_max < idx_min and frame_count - last_jumped_frame > JUMP_INTERVAL:
        print("JUMP")
        last_jumped_frame = frame_count


      #print(boxes.shape, boxes)
      #print(scores.shape,scores)
      #print(classes.shape,classes)
      #print(num_detections)


      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
#          image_np,
          image,
          np.array(faces),
          np.array(face_classes).astype(np.int32),
          np.array(face_scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=3)

      out.write(image)

      cv2.imshow("tensorflow based (%d, %d)" % (1024, 768), image)
      k = cv2.waitKey(1) & 0xff
      if k == ord('q') or k == 27:
          break
      if k == ord('p'):
          time.sleep(10)


    cap.release()
    out.release()
    out_file.close()
