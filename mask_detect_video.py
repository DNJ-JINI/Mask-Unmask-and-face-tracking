##from keras.models import load_model
import os
import numpy as np
import tensorflow as tf
import cv2
import datetime
import pandas as pd
import shutil
source=cv2.VideoCapture(0)
cf=0
if os.path.exists("./Frame/"):
    shutil.rmtree('./Frame/', ignore_errors=False, onerror=None)
if not os.path.exists('./Frame/'):
    os.makedirs('./Frame/')
if os.path.exists("./Crop_Frame/"):
    shutil.rmtree('./Crop_Frame/', ignore_errors=False, onerror=None)
if not os.path.exists('./Crop_Frame/'):
    os.makedirs('./Crop_Frame/')
if os.path.exists("vnew.csv"):
    os.remove("vnew.csv")


class FaceDetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')


    def processimage(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))
        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()


def crop_face(fl_name):
    
    classes ={1:"WITHOUT MASK",0:"MASK"}
    dict={1:(0,255,0),0:(255,0,255)}
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    img = cv2.imread(fl_name,1)
    boxes, scores, classes, num = fdapi.processimage(img)
    height = img.shape[0]
    img1=img.copy()
    id=1
    
    
 

    
    # Visualization of the results of a detection.
    for i in range(len(boxes)):
         if scores[i] > threshold:
            box = boxes[i]
            lab = classes[i]-1
            
            if (lab==0):
                mask='Yes'
                m="MASK"
            else:
                mask='No'
                m="WITHOUT MASK"

            cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),dict[lab],2)
            crop_img = img1[box[0]:box[2],box[1]:box[3]]
            file1 = open("vnew.csv","a") 
            
            cv2.imwrite('./Crop_Frame/{}.jpg'.format(id), crop_img)
            csv_flname=('./Crop_Frame/{}_{}.jpg'.format(fl_name.split(os.sep)[-1],id))
            file1.writelines(csv_flname)
            
            file1.writelines(",")
            file1.writelines(mask)
            file1.writelines(",")
            file1.writelines(str(scores[i]))
            file1.writelines('\n')
                
                
            
            file1.close()
            id=id+1

            y = box[0] - 15 if box[0] - 15 > 15 else box[0] + 15
                                
            cv2.putText(img,m,(box[1], y),cv2.FONT_HERSHEY_SIMPLEX,0.5,dict[lab],2)

    key = cv2.waitKey(8)
    return img
if __name__ == "__main__":

    model_path = './frozen_inference_graph.pb'
    fdapi = FaceDetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    img_dir = './Frame'
    while(True):
         ret,img=source.read()
         nm='./Frame' + str(cf) + '.jpg'
         cv2.imwrite('./Frame/' + str(cf) + '.jpg',img)
         cf =cf+ 1
         if cv2.waitKey(5000):
             break
         
    cv2.destroyAllWindows()
    source.release()
        
    for fl in os.listdir(img_dir):
        print(fl)
        fl_name = img_dir+"/" +fl
        img = crop_face(fl_name)
        cv2.imshow('img',img)
        
        
    
   
        
        
        
        
        






