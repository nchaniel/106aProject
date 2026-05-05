# CLAUDE.md

# Current Code
This code uses a real sense camera attached to an arm, the arm moves around a completed "dish" this is a constructed dish that we want to recreate. The arm moves in 2 circles of 20, waypoints taking images at each waypoint in addition to a top down image, this is 41 pictures in total.

The segment_bath.py, takes the images saved in caputred_images and runs a yolo + sam2 pipeline, creating masks for each image, and saving to its own folder. I think having this work live would be really interesting, although we still plan on using the photos to create waypoints it would be good to show in our demo the masking and image detection working live, on a live stream. I have the yolo + masking code working in segment_live_code, i want to integrate that with the arm circler, so while it runs it broadcasts a live stream, from the realsense camera, showing yolo detection and sam2 masking side by side. While still capturing the images in captured_images.

# Considerations
The yolo + sam2 masking runs really slow ~0.5 fps, is there a way to optimize this, currently using a custom yolo model, best.pt and sam2 tiny