#!/usr/bin/env python3

import cv2

import numpy as np

from cvl.dataset import OnlineTrackingBenchmark
from cvl.trackers import NCCTracker, MOSSETracker

dataset_path = "../data/Mini-OTB"

SHOW_TRACKING = True
SEQUENCE_IDX = 4

PATCH_SCALING_FACTOR = 1

if __name__ == "__main__":

    dataset = OnlineTrackingBenchmark(dataset_path)

    a_seq = dataset[SEQUENCE_IDX]

    if SHOW_TRACKING:
        cv2.namedWindow("tracker")

    #tracker = NCCTracker()
    tracker = MOSSETracker(std=10, learning_rate=0.1)
    #resp = 0.0
    #norm_patch = np.array(0.0)

    for frame_idx, frame in enumerate(a_seq):
        print(f"{frame_idx} / {len(a_seq)}")
        image_color = frame['image']
        image = np.sum(image_color, 2) / 3

        if frame_idx == 0:
            bbox = frame['bounding_box']
            #bbox.xpos -= bbox.width // PATCH_SCALING_FACTOR
            #bbox.width = int(bbox.width * PATCH_SCALING_FACTOR)
            #bbox.ypos -= bbox.height // PATCH_SCALING_FACTOR
            #bbox.height = int(bbox.height * PATCH_SCALING_FACTOR)
            
            if bbox.width % 2 == 0:
                bbox.width += 1

            if bbox.height % 2 == 0:
                bbox.height += 1

            tracker.start(image, bbox)
        else:
            resp = tracker.detect(image)
            norm_patch = tracker.update(image)

        if SHOW_TRACKING and frame_idx > 0:
            bbox = tracker.region
            pt0 = (bbox.xpos, bbox.ypos)
            pt1 = (bbox.xpos + bbox.width, bbox.ypos + bbox.height)
            image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_color, pt0, pt1, color=(0, 255, 0), thickness=3)
            
            # Show full color image with bounding box
            cv2.imshow("tracker", image_color)
            
            # Show patch used for update
            #cv2.imshow("norm_patch", norm_patch)
            
            # Response pattern (should be gaussian ish)
            #resp = np.abs(resp)
            #cv2.imshow("tracker", (resp - np.min(resp)) / (np.max(resp) - np.min(resp)))
            
            cv2.waitKey(0)
