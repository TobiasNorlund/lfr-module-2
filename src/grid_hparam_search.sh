#!/bin/bash

bbox_enlarge_factors=(1.0 1.5 2.0 2.5 3.0)
learning_rates=(0.0 0.01 0.05 0.1 0.2 0.5 1.0)
stds=(1.0 2.0 3.0 5.0 10.0)

echo -e "bbox_enlarge_factor\tlearning_rate\tstd\tmean_average_iou"
for bbox_enlarge_factor in "${bbox_enlarge_factors[@]}"
do
    for learning_rate in "${learning_rates[@]}"
    do
        for std in "${stds[@]}"
        do
            set -x
            mean_avg_iou=$(python evaluate.py --tracker MOSSE --bbox-enlarge-factor $bbox_enlarge_factor --learning-rate $learning_rate --std $std)
            set +x
            echo -e "$bbox_enlarge_factor\t$learning_rate\t$std\t$mean_avg_iou"
        done
    done
done