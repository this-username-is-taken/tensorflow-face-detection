#!/bin/bash

while true; do
    cd /home/nvidia/Projects/tensorflow-face-detection
    echo "Restarted at $(date +%Y%m%d_%H%M%S)" >> log
    python inference_leapbox.py
done
