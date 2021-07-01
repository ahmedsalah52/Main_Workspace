
#/home/ahmed/anaconda3/envs/tracker-gpu/lib/python3.7/
#"/usr/bin/python3"


from os import name
import sys
#sys.path.append('/home/ahmed/ITI_ROS_WS/src/tracking_pkg/tracking_pkg')

#sys.path.append('/home/ahmed/anaconda3/envs/tracker-gpu/lib')
#print(sys.path)
import rclpy
import cv2
import math

from rclpy.node import Node 
from example_interfaces.msg import String ,Int64
from example_interfaces.srv import SetBool
from msgs.msg import Vector

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
import torch
import cv2

import torch
from pred import Prediction
from shapely.geometry import Polygon


#ros imports







class my_node(Node):
    def __init__(self):
        super().__init__("node")
        #configs

        #which cam you want to use , do not forget to check the topic
        Mono_cam = False
        Zed_cam = not Mono_cam
        
        #objects we are interested in
        self.objects_of_interest = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
         'stop sign', 'parking meter', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe']
        
        
        #according to https://support.stereolabs.com/hc/en-us/articles/360007395634-What-is-the-camera-focal-length-and-field-of-view-
        #hd720 res
        #FOV
        self.camera_H_angle =  104.0
        self.camera_V_angle = 72.0
      
        #after how many steps you want to predict
        self.num_of_timesteps_prediction = 50


        #init
        self.depth_array  = np.array(np.zeros([360, 640]), dtype=np.float32)
        self.plan_view_image = np.zeros([720, 1280,3])
        self.point_cloud_plan = np.zeros([720, 1280])
        self.camera_x_pos_from_car = 0.0 #in meters , 0 means in the center 
        self.pixels_per_m = self.plan_view_image.shape[0]/20 #in plan view , 20 is the farest object to be plotted
        self.car_dims = list((1,2)) #in meters (x,y) planview
        self.car_plan_view = np.zeros([720, 1280,3])
        start = (int(self.car_plan_view.shape[1]/2 -(self.pixels_per_m *self.car_dims[0]/2)),self.car_plan_view.shape[0]-int(self.car_dims[1]*self.pixels_per_m))
        end =  (int(self.pixels_per_m *self.car_dims[0]/2+self.car_plan_view.shape[1]/2),self.car_plan_view.shape[0])
        
        cv2.rectangle(self.car_plan_view, start ,end , (255,255,255), 10)
        self.avg_fps = 0.0
        self.fps_counter = 0.0 

        

        start = (int(self.car_plan_view.shape[1]/2 -(self.pixels_per_m *self.camera_x_pos_from_car/2)),self.car_plan_view.shape[0]-int(self.car_dims[1]*self.pixels_per_m))
        line_length = self.pixels_per_m *30
        zero_angle = 180
        end =  (int(start[0]+(line_length * math.sin(math.radians(-zero_angle+(self.camera_H_angle/2))))),int(start[1]+(line_length * math.cos(math.radians(-zero_angle+(self.camera_H_angle/2))))))
        cv2.line(self.car_plan_view, start, end, (255,255,255), 1)
        
        end =  (int(start[0]+(line_length * math.sin(math.radians(-zero_angle-(self.camera_H_angle/2))))),int(start[1]+(line_length * math.cos(math.radians(-zero_angle-(self.camera_H_angle/2))))))

        cv2.line(self.car_plan_view, start, end, (255,255,255), 1)
        
        
        cv2.line(self.car_plan_view, start, (start[0],0), (0.2,0.2,0.2), 1)

        y = int(start[1])

        while(1):
            x_start = int(0)
            x_end = int(self.car_plan_view.shape[1]-1)
            start_point = ((x_start , int(y)))
            end_point = ((x_end , int(y)))
            cv2.line(self.car_plan_view, start_point, end_point, (0.2,0.2,0.2), 1)

            y = y - self.pixels_per_m
            if y <0:
                break
        
        if Zed_cam:
            self.create_subscription(Image,"/zed2/zed_node/depth/depth_registered",self.sub_call_depth,rclpy.qos.qos_profile_sensor_data) 
            self.create_subscription(Image,"/zed2/zed_node/rgb/image_rect_color",self.sub_call_image,rclpy.qos.qos_profile_sensor_data) 
            self.create_subscription(Vector,"/points_array",self.points_array_call,rclpy.qos.qos_profile_sensor_data) 

        else:
            self.create_subscription(Image,"depth_image",self.sub_call_depth,rclpy.qos.qos_profile_sensor_data) 
            self.create_subscription(Image,"original_image",self.sub_call_image,rclpy.qos.qos_profile_sensor_data) 
        
        


 #       self.create_subscription(Image,"/zed2/zed_node/depth/depth_registered",self.sub_call_depth,rclpy.qos.qos_profile_sensor_data) 
#        self.create_subscription(Image,"/zed2/zed_node/rgb/image_rect_color",self.sub_call_image,rclpy.qos.qos_profile_sensor_data) 

        self.get_logger().info("Subscriber  is started now")
        self.bridge = CvBridge()

        self.tracker_init()

    def points_array_call(self,msg):
        print("point")
        data = msg.data
       
        
        plan_view =  np.zeros([720, 1280])
        
        for y , x in enumerate(data): #recieve x in y and y in x because x in image is right and left, and y in points recieved are left and right and the same with forward and backward
            x_pixels = (float(x)/100) * self.pixels_per_m
            y_pixels = (float(y)/100) * self.pixels_per_m
    
            if not(y_pixels == 0 or x_pixels == 0):
                plan_view[ int((plan_view.shape[0])-y_pixels-int(self.car_dims[1]*self.pixels_per_m)), int((plan_view.shape[1]/2)-x_pixels )] = True
        #self.pixels_per_m
        #cv2.imshow('Plan View',plan_view )
        # press q to quit
        #if cv2.waitKey(1) == ord('q'):
        #    cv2.destroyAllWindows()
        self.point_cloud_plan = plan_view



    def sub_call_image(self,msg):
        
       
        image = self.bridge.imgmsg_to_cv2(msg)

        img = image[:,:,0:3]
        #self.get_logger().info(str(img.shape))
        img = np.resize(img,(360, 640, 3))
        #self.get_logger().info(str(img.shape))
        #vis = np.concatenate((img1, img2), axis=0)
        self.track(img)
        self.plan_view_image[self.point_cloud_plan == True] = (255,255,0)
        cv2.imshow('plan view', self.plan_view_image )
        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
        
        
    def sub_call_depth(self,msg):
	
       
        depth_image = self.bridge.imgmsg_to_cv2(msg)
        

        depth_array = np.array(depth_image, dtype=np.float32)
       
        #self.get_logger().info('depth array {} , {}'.format(msg.height,msg.width))
        self.depth_array = depth_array

        
        
	
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
        self.tracker = Tracker(self.metric ,max_iou_distance=0.7, max_age=30, n_init=3)

        self.yolo5 = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # size 640 from fast yolov5s ,yolov5m yolov5l yolov5x , size 1280 from fast yolov5s6 ,yolov5m6,yolov5l6,yolov5x6

        self.prediction_manager = Prediction(sudden_move_thres = 150 , age =2)
        self.get_logger().info("initiatingdone ")

    

        
        
    def track(self,input_frame):
        self.get_logger().info("tracking")
        self.plan_view_image = self.car_plan_view.copy()


        img = input_frame
        fps = 0.0
        count = 0 
        angle = 0.0
        if img is None:
            self.get_logger().warn("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                pass
            else: 
                return False
        
        depth_array = self.depth_array.copy()

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        #img_in = tf.expand_dims(img_in, 0)
        #img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        results = self.yolo5(img_in)  
        
        output = results.pandas().xyxy[0]
        
        output = output.to_numpy()
                        #xmin     ymin      xmax     ymax     confidence   class          name
              #  0   582.500000  143.750  618.5000  209.000    0.782715      0           person

        #print(output.T)
        output = output.T

        boxes= []
        scores= []
        names = []

        boxes_ = output[0:4].T
        scores_ = output[4]
        classes_ = output[5]
        names_  = output[6]


        


        for i in range(len(names_)):
            if ((scores_[i] > 0.5) and (names_[i] in self.objects_of_interest)):
                boxes.append(boxes_[i])
                scores.append(scores_[i])
                names.append(names_[i])

      
        
        converted_boxes = convert_boxes(img, boxes)
        #print(converted_boxes)
        features = self.encoder(img, converted_boxes)    
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores, names, features)]
        
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
        #print("sec ",scores)

        my_t1 = time.time()
        #objects_masks = np.array(np.zeros([360, 640]), dtype=np.float32)
        for track in detections:
            bbox = track.to_tlbr()
            #tracked_obj_id = int(track.track_id)
            depth_array[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])] += 100
            


        depth_array -= 100
       
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            
            class_name = track.get_class()
            #s = str(scores_[counter])+"-"+str(names[counter])
            tracked_obj_id = int(track.track_id)
            color = colors[int(tracked_obj_id) % len(colors)]
            color = [i * 255 for i in color]
            obj_color = color
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(tracked_obj_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(img, class_name + "-" + str(tracked_obj_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            
            
            center = list((int(bbox[0]+((bbox[2]-bbox[0])/2)),int(bbox[1]+((bbox[3]-bbox[1])/2))))
            #print("center {}".format(center))
            if center[1] >= 360:
                center[1] = 359
            if center[0] >= 640:
                center[0] = 639



            object_boundry = depth_array[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])] 
            


            min_distance_position=list((1000,1000))
            distance = 1000000000.0
            #column_min_distaces = np.zeros(object_boundry.shape[0])
            start_box_x = bbox[0]
            start_box_y = bbox[1]
            min_distance_row = 0
            for i ,row in enumerate(object_boundry):
                for j,d in enumerate(row):
                    if d > 0:
                        if d< distance:
                        
                            distance = d
                            min_distance_row = row
                            #print(min_distance_row)
                            min_distance_row_num = i

                            min_distance_position = list((int(j+start_box_x),int(i+start_box_y)))
           

            if(distance>0) and (distance<50):
                text = "{:.2f}".format(distance)
                cv2.putText(img, text, center, 0, 0.75, (0,255,0), 2)

                obj_color = (obj_color[0]/255,obj_color[1]/255,obj_color[2]/255)

                #center_coordinates = self.point_from_img_to_planview(img.shape,self.plan_view_image.shape,center,distance)
                radius = 5
                
                
                text =  class_name + "-" + str(tracked_obj_id)
               
                point_coor = self.point_from_img_to_planview(img.shape,self.plan_view_image.shape,min_distance_position,distance)

                #cv2.circle(self.plan_view_image, point_coor, radius,obj_color, 2)



            
                self.prediction_manager.set_object(point_coor,tracked_obj_id,class_name)
                
                state = self.prediction_manager.get_object_current_states(tracked_obj_id)
                current_pos = (int(state[0]),int(state[2]))
                
                
                cv2.circle(self.plan_view_image, current_pos, radius,obj_color, 2)
                cv2.putText(self.plan_view_image, text, (current_pos[0]+10,current_pos[1]), 0, 0.75, obj_color, 2)
                for i in range(self.num_of_timesteps_prediction):
                    state = self.prediction_manager.get_object_next_states(tracked_obj_id)
                    next_pos = (int(state[0]),int(state[2]))
                    cv2.arrowedLine(self.plan_view_image, current_pos, next_pos,(0,0,255), 3) 

            
               
                y = start_box_y +min_distance_row_num # center[1]
                
                if min_distance_row_num == 0:
                    y = center[1]
                    print("center y ")
                points = []
                
                for i,dist in enumerate(min_distance_row):
                    x = i + start_box_x
                    if x >= 640:
                        x = 639

                    if dist >distance and dist < (distance+2):
                        point_coor = self.point_from_img_to_planview(img.shape,self.plan_view_image.shape,(x,y),dist)
                        points.append(point_coor)

                    
                    
                points = np.array(points,dtype=np.int32)

                cv2.polylines(self.plan_view_image, [points], 
                    False, (0,0.3,0), 
                    2)
            #print(color)

            #self.get_logger().info(text)
        ### UNCOMMENT BELOW IF YOU WANT CONSTANTLY CHANGING YOLO DETECTIONS TO BE SHOWN ON SCREEN
        for det in detections:
            bbox = det.to_tlbr() 
            cv2.rectangle(depth_array,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

        # print fps on screen 
        print(time.time() -my_t1)

        self.prediction_manager.life_iteration()
        fps  = ( fps + (1./(time.time()-t1)) ) / 2

        self.avg_fps += fps
        self.fps_counter +=1
        avg  = self.avg_fps /self.fps_counter
        cv2.putText(img, "FPS: {:.2f} AVG FPS: {:.2f}".format(fps,avg), (0, 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(0,0,255), 2)
        cv2.imshow('output', img)
        cv2.imshow('depth',cv2.convertScaleAbs(depth_array))
       
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

    def area(self,rect):
        l = rect[2] - rect[0]
        w = rect[3] - rect[1]
        return l*w

    def isRectangleOverlap(self, R1, R2):
        if (R1[0]>=R2[2]) or (R1[2]<=R2[0]) or (R1[3]<=R2[1]) or (R1[1]>=R2[3]):
            return False
        else:
            return True

    def point_from_img_to_planview(self,img_shape,planview_img_shape,point,distance):
        angle = ((img_shape[1]/2) - point[0]) * (self.camera_H_angle/img_shape[1])
        radians = math.radians(angle)

        

        Horizontal_distance = math.tan(radians) * distance  * self.pixels_per_m
        center_depth = distance * self.pixels_per_m 
       
        center_coordinates = (int(planview_img_shape[1]/2 -Horizontal_distance) 
        , planview_img_shape[0]- int((center_depth)+(self.car_dims[1] * self.pixels_per_m)))


        return center_coordinates




  
def main(args = None):
    rclpy.init(args=None)
    node = my_node()
    rclpy.spin(node)

    rclpy.shutdown()



if __name__ == "__main__":
    main()
