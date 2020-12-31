from pathlib import Path
import numpy as np
from cv2 import resize, cvtColor, COLOR_BGR2RGB
import tensorflow as tf

class Predictor:

    def __init__(self, model_path: str):
        self.__initModel(model_path)
        

    class ModelDoesNotExist(Exception): pass
    def __initModel(self, model_path):
        if not Path(model_path).exists(): raise self.ModelDoesNotExist('Check model path') 

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]['shape']
        self.input_size = tuple(self.input_shape[1:3])

        self.input_index = self.input_details[0]['index']
    
    def imageFilter(self, frame, invertBR=False):
        if invertBR: frame = cvtColor(frame, COLOR_BGR2RGB)
        image_data = resize(frame, self.input_size)
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        
        return image_data

    def forward_pass(self, image):
        self.interpreter.set_tensor(self.input_index, image)
        self.interpreter.invoke()
        pred = [self.interpreter.get_tensor(self.output_details[i]['index']) for i in range(len(self.output_details))]
        return pred

    def filter_boxes(self, box_xywh, scores, score_threshold=0.4, input_shape = tf.constant([416,416]), log=False):
        scores_max = tf.math.reduce_max(scores, axis=-1)
        mask = scores_max >= score_threshold
        class_boxes = tf.boolean_mask(box_xywh, mask)
        pred_conf = tf.boolean_mask(scores, mask)
        class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
        pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])
        box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)
        input_shape = tf.cast(input_shape, dtype=tf.float32)
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        box_mins = (box_yx - (box_hw / 2.))/ input_shape
        box_maxes = (box_yx + (box_hw / 2.))/ input_shape
        boxes = tf.concat([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ], axis=-1)
        
        return (boxes, pred_conf)

    def non_max_supression(self, boxes, pred_conf, IOU=0.4, SCORE = 0.25):

        classes, valid_detection = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=IOU,
            score_threshold=SCORE)[2:]

        return classes, valid_detection


    def getPredictions(self, frame):
        frame = self.imageFilter(frame)
        pred = self.forward_pass(frame)
        boxes, pred_conf = self.filter_boxes(pred[1], pred[0], 
                                score_threshold=0.25, input_shape=tf.constant([self.input_size]))
        classes, valid_detection = self.non_max_supression(boxes, pred_conf)

        return classes.numpy()[0, :valid_detection.numpy()[0]].astype(int)