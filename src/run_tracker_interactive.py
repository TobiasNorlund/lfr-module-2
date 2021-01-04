#!/usr/bin/env python3

import cv2
import numpy as np
from cvl.dataset import OnlineTrackingBenchmark
import ncc
import mosse
import trackers

DATASET_PATH = "../data/Mini-OTB"

def _scale_factor(a):
    amin = np.min(a)
    amax = np.max(a)
    if amax-amin <= 1e-3: 
        scale_factor = 1.1*amax
    else:
        scale_factor = amax-amin
    return scale_factor

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser.add_argument("--seq-idx", type=int, default=0)
    parser.add_argument("--bbox-enlarge-factor", type=float, default=1.0)
    parser.add_argument("--visualize-filter", type=bool, default=False)
    trackers.add_tracker_args(parser)
    args = parser.parse_args()

    visualize_filter = args.visualize_filter
    tracker = trackers.get_tracker(args)

    if visualize_filter: 
        assert hasattr(tracker, "get_filter"), "The specified tracker must have a get_tracker function implemented"

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

        if visualize_filter:
            wind, filt, resp = tracker.get_filter(frame['image'])
            filt_height, filt_width = filt.shape

            # Normalize image
            wind = (wind-np.min(wind)) / _scale_factor(wind)
            filt = (filt-np.min(filt)) / _scale_factor(filt)
            resp = (resp-np.min(resp)) / _scale_factor(resp)

            # Concatenate the window, filter and response 
            conc_img = np.float32(np.concatenate([wind, filt, resp], axis=1))
            scale = 5
            conc_img = cv2.resize(conc_img, dsize=(3*scale*filt_width, scale*filt_height))

            # Add descriptive text.
            conc_img = cv2.putText(conc_img, "Window", (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
            conc_img = cv2.putText(conc_img, "Filter", (5+scale*filt_width, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
            conc_img = cv2.putText(conc_img, "Response", (5+2*scale*filt_width, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
    
            cv2.imshow("filter", conc_img)

        cv2.waitKey(0)