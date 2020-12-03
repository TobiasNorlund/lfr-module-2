#!/usr/bin/env python3

import cv2

import numpy as np

from cvl.dataset import OnlineTrackingBenchmark
from cvl.trackers import NCCTracker, MOSSETracker

dataset_path = "../data/Mini-OTB"

SHOW_TRACKING = True
SEQUENCE_IDX = 4

if __name__ == "__main__":

    dataset = OnlineTrackingBenchmark(dataset_path)

    a_seq = dataset[SEQUENCE_IDX]

    if SHOW_TRACKING:
        cv2.namedWindow("tracker")

    #tracker = NCCTracker()
    tracker = MOSSETracker(std=20, learning_rate=0.5)
    resp = 0.0

    for frame_idx, frame in enumerate(a_seq):
        print(f"{frame_idx} / {len(a_seq)}")
        image_color = frame['image']
        image = np.sum(image_color, 2) / 3

        if frame_idx == 0:
            bbox = frame['bounding_box']
            if bbox.width % 2 == 0:
                bbox.width += 1

            if bbox.height % 2 == 0:
                bbox.height += 1

            tracker.start(image, bbox)
        else:
            resp = tracker.detect(image)
            tracker.update(image)

        if SHOW_TRACKING:
            bbox = tracker.region
            pt0 = (bbox.xpos, bbox.ypos)
            pt1 = (bbox.xpos + bbox.width, bbox.ypos + bbox.height)
            image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_color, pt0, pt1, color=(0, 255, 0), thickness=3)
            cv2.imshow("tracker", image_color)
            #cv2.imshow("tracker", (image_color.mean(axis=2) + np.real(resp) * 1e6) / 255)
            cv2.waitKey(0)
