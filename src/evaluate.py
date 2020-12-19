import ncc
import mosse
import numpy as np
import trackers
from cvl.dataset import OnlineTrackingBenchmark
from multiprocessing import Pool
from functools import partial


DATASET_PATH = "../data/Mini-OTB"


def run_tracker_on_sequence(tracker, sequence, bbox_enlarge_factor):
    """
    Runs a tracker on a sequence and returns the resulting bounding boxes for each frame
    """
    bboxes = []
    for frame_idx, frame in enumerate(sequence):
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
            bbox = tracker.detect(image_color)
            tracker.update(image_color)
        bboxes.append(bbox.rescale(1/bbox_enlarge_factor))
    return bboxes


def evaluate(tracker, bbox_enlarge_factor):
    """
    Evaluates mean average IoU = Average IoU over all frames in a sequence, and mean over all sequences
    """
    otb = OnlineTrackingBenchmark(DATASET_PATH)
    per_seq_mean_iou = []
    
    for sequence_idx, sequence in enumerate(otb):
        bboxes = run_tracker_on_sequence(tracker, sequence, bbox_enlarge_factor)
        mean_iou = otb.calculate_mean_iou(sequence_idx, bboxes)
        per_seq_mean_iou.append(mean_iou) 
        print(mean_iou)
    return np.mean(per_seq_mean_iou)


def _evaluate_sequence(seq_idx, args):
    tracker = trackers.get_tracker(args)
    otb = OnlineTrackingBenchmark(DATASET_PATH)
    bboxes = run_tracker_on_sequence(tracker, otb[seq_idx], args.bbox_enlarge_factor)
    mean_iou = otb.calculate_mean_iou(seq_idx, bboxes)
    return mean_iou

def evaluate_parallel(args):

    f = partial(_evaluate_sequence, args=args)
    
    num_sequences = len(OnlineTrackingBenchmark(DATASET_PATH))
    with Pool() as p:
        per_seq_mean_iou = p.map(f, list(range(num_sequences)))
    
    return np.mean(per_seq_mean_iou)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser.add_argument("--bbox-enlarge-factor", type=float, default=1.0)
    trackers.add_tracker_args(parser)
    args = parser.parse_args()
    
    mean_average_iou = evaluate_parallel(args)
    print(mean_average_iou)
