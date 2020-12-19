#!/usr/bin/env python3

import cv2
import numpy as np
from cvl.dataset import OnlineTrackingBenchmark
import ncc
import mosse
import trackers

DATASET_PATH = "../data/Mini-OTB"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser.add_argument("--seq-idx", type=int, default=0)
    parser.add_argument("--bbox-enlarge-factor", type=float, default=1.0)
    trackers.add_tracker_args(parser)
    args = parser.parse_args()

    tracker = trackers.get_tracker(args)

    dataset = OnlineTrackingBenchmark(DATASET_PATH)
    a_seq = dataset[args.seq_idx]

    cv2.namedWindow("tracker")

    for frame_idx, frame in enumerate(a_seq):
        print(f"{frame_idx} / {len(a_seq)}")
        image_color = frame['image']

        if frame_idx == 0:
            bbox = frame['bounding_box']
            bbox = bbox.rescale(args.bbox_enlarge_factor)
            
            if bbox.width % 2 == 0:
                bbox.width += 1

            if bbox.height % 2 == 0:
                bbox.height += 1

            tracker.start(image_color, bbox)
        else:
            tracker.detect(image_color)
            tracker.update(image_color)

        bbox = tracker.region
        pt0 = (bbox.xpos, bbox.ypos)
        pt1 = (bbox.xpos + bbox.width, bbox.ypos + bbox.height)
        image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
        cv2.rectangle(image_color, pt0, pt1, color=(0, 255, 0), thickness=3)
        cv2.imshow("tracker", image_color)
        cv2.waitKey(0)
