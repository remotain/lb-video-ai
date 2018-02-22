from __future__ import division, print_function, absolute_import

import argparse
import os, sys, urllib, tarfile
from datetime import timedelta
from datetime import datetime
import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from application_util.image_viewer import ImageViewer
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

import PIL.Image as Image

import tensorflow as tf

import tools.generate_detections as generate_detections

# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append('../tf-models/research/object_detection/')
# ## Object detection imports
# Here are the imports from the object detection module.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from timeit import default_timer as timer



def run(video_source, path_object_model, path_encoder_model, path_labels, 
        min_score_thresh, nms_max_overlap, max_cosine_distance, 
        nn_budget, display, time_profile):

    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    video_source : str
        Path to the video source to process.
    path_object_model : str
        Path to object recognition model.
    path_encoder_model : str
        Path to encoder model.
    path_labels : str
        Path to object labels.
    min_score_thresh : float
        Detection confidence threshold. Disregard 
        all detections that have a confidence lower than this value
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.
    time_profile : bool
        If True, Show timing informations.
    """
    
    def timeit(method):

        def timed(*args, **kw):
            ts = timer()
            result = method(*args, **kw)
            te = timer()
            
            if time_profile:
                print('%r %2.3f sec' % (method.__name__, te-ts))
            return result

        return timed
    
    # Open video stream
    cap = cv2.VideoCapture(video_source)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Deep SORT stuff
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    
    if not os.path.exists(path_encoder_model):
        print("%s: No such file or directory" % path_encoder_model)
        sys.exit(1)        
    encoder = generate_detections.create_box_encoder(path_encoder_model)
        
    # Object detection
    
    # ## Check if model exist otherwise download it
    OBJECT_MODEL_PATH = os.path.join(path_object_model, '')
    OBJECT_MODEL_FILE = os.path.join(OBJECT_MODEL_PATH, 'frozen_inference_graph.pb')
    
    if not os.path.exists(OBJECT_MODEL_PATH):
        
        DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

        DOWNLOAD_FILE = str.split(OBJECT_MODEL_PATH, '/')[-2] + '.tar.gz'
       
        DOWNLOAD_TO = os.path.join(str.split(OBJECT_MODEL_PATH, '/')[0], '')
        
        print('Model \"%s\" not on disk' % OBJECT_MODEL_PATH)
        print('Download it from %s' % (DOWNLOAD_BASE + DOWNLOAD_FILE))

        opener = urllib.request.URLopener()
        opener.retrieve( os.path.join(DOWNLOAD_BASE,DOWNLOAD_FILE), 
                         os.path.join(DOWNLOAD_TO,DOWNLOAD_FILE))

        # Extract tar the model from the tar file
        print('Extract frozen tensorflow model')
        tar_file = tarfile.open(os.path.join(DOWNLOAD_TO,DOWNLOAD_FILE))
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, DOWNLOAD_TO)
    
    # ## Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(OBJECT_MODEL_FILE, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        with tf.Session() as sess:
            # Get handles to input and output tensors
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
            tensor_list = [detection_boxes, detection_scores, detection_classes, num_detections]
                
    # ## Loading label map
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `airplane`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    if not os.path.exists(path_labels):
        print("%s: No such file or directory" % path_labels)
        sys.exit(1)
    
    label_map = label_map_util.load_labelmap(path_labels)
    categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=90,
            use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    #
    # ## Select some category to display
    # 1 : person
    # 2 : bycicle
    # 3 : car
    # 4 : motorcicle
    # 6 : bus
    # 8 : truck
    #idx_to_keep = [1,2,3,4,6,8]
    #category_index = { i: category_index[i] for i in idx_to_keep}
          
    # end of initialization
             
    # # Detection
    @timeit
    def object_detection(image, graph):
        
        (boxes, scores, classes, num) = sess.run(
            tensor_list,
            feed_dict={image_tensor: np.expand_dims(image, 0)})
        
        mask = scores > min_score_thresh
        classes = classes[mask]
        boxes = boxes[mask]
        scores = scores[mask]

        return (classes, boxes, scores)
    
    @timeit
    def extract_features(image, boxes):
            
        image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
        im_width, im_height = image_pil.size
        detections = []
    
        for box in boxes:
        
            ymin, xmin, ymax, xmax = box
            (left, right, bottom, top) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
        
            detections.append(np.array([left, bottom, right-left, top-bottom]))
            #scores.append(score)
   
        detections = np.array(detections)
    
        features = encoder(image, detections)
    
        detections = [Detection(bbox, 1.0, feature) 
            for bbox, feature in zip(detections, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
    
        scores = np.array([d.confidence for d in detections])
    
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
    
        detections = [detections[i] for i in indices]
    
        return detections
  
    @timeit
    def tracking(detections):
        tracker.predict()
        tracker.update(detections)
        return tracker
      
  
    @timeit
    def frame_callback():                            
        ret, frame_np = cap.read()
        
        # Resize frame of video to 1/4 size for faster face recognition processing
        #frame_np = cv2.resize(frame_np, (0, 0), fx=0.25, fy=0.25)
        
        # Skip bad read frames
        if not ret:
            return
        
        # Do things here        
        if time_profile:
            t_obj_start = timer()
            
        # Actual detection.
        tf_classes, tf_boxes, tf_scores = object_detection(frame_np, detection_graph)
    
        # Do things here        
        if time_profile:
            t_obj_stop = timer()
            t_feat_start = timer()
          
        detections = extract_features(frame_np, tf_boxes)
            
        # Update tracker.
        tracker = tracking(detections)
           
        for track,tf_class,tf_score in zip(tracker.tracks, 
                                            tf_classes, 
                                            tf_scores):
            
            bbox = track.to_tlbr()
            
            if display:
                
                h, w, _ = frame_np.shape
                thick = int((h + w) / 300.)
                
                cv2.rectangle(frame_np, 
                    ( int(bbox[0]), int(bbox[1]) ), 
                    ( int(bbox[2]), int(bbox[3]) ),
                    visualization.create_unique_color_uchar(track.track_id, hue_step=0.41), thick)
                    #(255,255,255), thick)
                    
                cv2.putText(frame_np,
                    str('id: %i, class: %s, score: %.2f' % (track.track_id, category_index[tf_class]['name'], tf_score )),
                    ( int(bbox[0]), int(bbox[1]) - 12), 0, 1e-3 * h, (255,0,0), int(thick/3))
              
                cv2.imshow('object detection', cv2.resize(frame_np,(800,450)))
                #cv2.imshow('object detection', frame_np)


    while True:
        
        print('Frame %i, %s' % (frame_count, datetime.now()))
         
        frame_callback()
        
        frame_count += 1
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break        
        
    cap.release()
    cv2.destroyAllWindows()
             
 
def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Video Deep SORT")
    parser.add_argument(
        "--video_source", help="Path to the video source to process.",
        default=0)
    parser.add_argument(
        "--path_object_model", help="Path to object recognition model.", 
        default="data/ssd_mobilenet_v1_coco_2017_11_17/", 
        choices=["data/ssd_mobilenet_v1_coco_2017_11_17/",
                "data/faster_rcnn_inception_v2_coco_2018_01_28/",
                "data/ssd_inception_v2_coco_2017_11_17/",
                "data/faster_rcnn_resnet101_coco_2018_01_28/"])
    parser.add_argument(
        "--path_encoder_model", help="Path to encoder model.", 
        default="data/networks/mars-small128.pb")
    parser.add_argument(
        "--path_labels", help="Path to object labels.", 
        default="models/research/object_detection/data/mscoco_label_map.pbtxt")
    parser.add_argument(
        "--min_score_thresh", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.5, type=float)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool)
    parser.add_argument(
        "--time_profile", help="Show timing informations",
        action='store_true')

    return parser.parse_args()
 
            
if __name__ == "__main__":
    
    args = parse_args()
    run(
        args.video_source, args.path_object_model, args.path_encoder_model,
        args.path_labels, args.min_score_thresh, args.nms_max_overlap,
        args.max_cosine_distance, args.nn_budget, args.display, args.time_profile)