import numpy as np
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment

def iou(bb_test, bb_gt):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Boxes are in format [x1, y1, x2, y2].
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + 
              (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o

class SORTTracker:
    def __init__(self, max_disappeared=15, iou_threshold=0.3):
        self.next_object_id = 0
        self.objects = OrderedDict() # {id: centroid (cx, cy)}
        self.object_rects = OrderedDict() # {id: [x1, y1, x2, y2]}
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold

    def register(self, rect):
        self.objects[self.next_object_id] = self._get_centroid(rect)
        self.object_rects[self.next_object_id] = rect
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.object_rects[object_id]
        del self.disappeared[object_id]

    def _get_centroid(self, rect):
        return (int((rect[0] + rect[2]) / 2.0), int((rect[1] + rect[3]) / 2.0))

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for rect in rects:
                self.register(rect)
        else:
            object_ids = list(self.objects.keys())
            tracked_rects = list(self.object_rects.values())
            
            # Compute Cost Matrix based on IoU distance (1 - IoU)
            iou_matrix = np.zeros((len(tracked_rects), len(rects)), dtype=np.float32)
            for t, trk in enumerate(tracked_rects):
                for d, det in enumerate(rects):
                    iou_matrix[t, d] = iou(trk, det)
                    
            cost_matrix = 1.0 - iou_matrix
            
            # Hungarian algorithm for optimal assignment
            matched_indices = linear_sum_assignment(cost_matrix)
            matched_indices = np.array(list(zip(*matched_indices)))
            
            unmatched_trackers = []
            for t, trk in enumerate(tracked_rects):
                if t not in matched_indices[:, 0]:
                    unmatched_trackers.append(t)
                    
            unmatched_detections = []
            for d, det in enumerate(rects):
                if d not in matched_indices[:, 1]:
                    unmatched_detections.append(d)
                    
            # Filter out matches with low IoU
            matches = []
            for m in matched_indices:
                if iou_matrix[m[0], m[1]] < self.iou_threshold:
                    unmatched_trackers.append(m[0])
                    unmatched_detections.append(m[1])
                else:
                    matches.append(m.reshape(1, 2))
                    
            if len(matches) == 0:
                matches = np.empty((0, 2), dtype=int)
            else:
                matches = np.concatenate(matches, axis=0)
                
            # Update matched trackers
            for m in matches:
                tracker_idx = m[0]
                detection_idx = m[1]
                object_id = object_ids[tracker_idx]
                rect = rects[detection_idx]
                
                self.object_rects[object_id] = rect
                self.objects[object_id] = self._get_centroid(rect)
                self.disappeared[object_id] = 0
                
            # Register new detections
            for d in unmatched_detections:
                self.register(rects[d])
                
            # Increment disappeared for unmatched trackers
            for t in unmatched_trackers:
                object_id = object_ids[t]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

        return self.objects
