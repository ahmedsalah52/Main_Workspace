
#/home/ahmed/anaconda3/envs/tracker-gpu/lib/python3.7/
#"/usr/bin/python3"
#import sys
#print(sys.path)

#libcublas.so.10
#libcudnn.so.7

import sys
sys.path.append('/home/ahmed/ITI_ROS_WS/src/tracking_pkg/tracking_pkg')

sys.path.append('/home/ahmed/anaconda3/envs/tracker-gpu/lib')
print(sys.path)
import rclpy
import cv2

from rclpy.node import Node 
from example_interfaces.msg import String ,Int64
from example_interfaces.srv import SetBool
from matplotlib import pyplot as plt
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import timeit

import time, random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import (YoloV3, YoloV3Tiny)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
#from PIL import Image


#ros imports



flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/test.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', './data/video/test_out.mp4', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')





class my_node(Node):
    def __init__(self):
        super().__init__("node")
        #self.image = np.zeros([240, 320, 3])
        self.create_subscription(Image,"/zed2/zed_node/rgb/image_rect_color",self.sub_call,rclpy.qos.qos_profile_sensor_data) #tracking
        self.get_logger().info("Subscriber  is started now")
        
        self.tracker_init()



    def sub_call(self,msg):
        self.bridge = CvBridge()

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgra8')

        img = image[:,:,0:3]
        self.get_logger().info(str(img.shape))
        img = np.resize(img,(360, 640, 3))
      
        self.track(img)

    def tracker_init(self):
        FLAGS(sys.argv)
        
        # Definition of the parameters
        self.max_cosine_distance = 0.5
        self.nn_budget = None
        self.nms_max_overlap = 1.0
        
        #initialize deep sort
        self.model_filename = 'model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(self.metric)

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        if FLAGS.tiny:
            self.yolo = YoloV3Tiny(classes=FLAGS.num_classes)

        else:
            self.yolo = YoloV3(classes=FLAGS.num_classes)

        self.yolo.load_weights(FLAGS.weights)
        logging.info('weights loaded')

        self.class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
        logging.info('classes loaded')
        self.get_logger().info("initiatingdone ")

    


        if FLAGS.output:
            # by default VideoCapture returns float instead of int
            
            self.list_file = open('detection.txt', 'w')
            self.frame_index = -1 
        
        
    def track(self,input_frame):
        self.get_logger().info("tracking")

        img = input_frame
        fps = 0.0
        count = 0 
        if img is None:
            self.get_logger().warn("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                pass
            else: 
                return False
        

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = self.yolo.predict(img_in)
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(self.class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = self.encoder(img, converted_boxes)    
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]
        
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]        

        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)

        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            
        ### UNCOMMENT BELOW IF YOU WANT CONSTANTLY CHANGING YOLO DETECTIONS TO BE SHOWN ON SCREEN
        #for det in detections:
        #    bbox = det.to_tlbr() 
        #    cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

        # print fps on screen 
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.imshow('output', img)
        '''
        if FLAGS.output:
            frame_index = self.frame_index + 1
            self.list_file.write(str(frame_index)+' ')
            if len(converted_boxes) != 0:
                for i in range(0,len(converted_boxes)):
                    self.list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1]) + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
            self.list_file.write('\n')
        '''


        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()

            return False

        #if FLAGS.ouput:
        #    self.list_file.close()
            #return True 
        





  
def main(args = None):
    rclpy.init(args=None)
    node = my_node()
    rclpy.spin(node)

    rclpy.shutdown()



if __name__ == "__main__":
    main()
