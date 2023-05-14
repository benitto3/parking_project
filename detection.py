import os
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path

# Configuration for the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class

# Filter only detected cars
def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car, motorcycle or truck, skip it
        if class_ids[i] in [3, 4, 6]:
            car_boxes.append(box)

    return np.array(car_boxes)

# Root directory of the project
ROOT_DIR = "."

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Value 0 or 1 for camera input
VIDEO_SOURCE = "sample/video/park.mp4"

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Load the video file we want to run detection on
video_capture = cv2.VideoCapture(VIDEO_SOURCE)

# Location of parking spaces
parked_car_boxes = None
# do we have free space
free_space = False
# list of all parking spaces
parking_list = []
# list of all detected free parking spaces
free_space_list = []
# frames of free parking space in a row
free_space_frames = 0
# total parking spaces number
total_car_boxes = 0
# total free parking spaces number
free_parking_space = 0

# Loop over each frame of video
while video_capture.isOpened():
    success, frame = video_capture.read()
    if not success:
        break
    # Convert the image from BGR to RGB color
    rgb_image = frame[:, :, ::-1]
    # Run the image through the Mask R-CNN model
    results = model.detect([rgb_image], verbose=0)
    # For only one image give the first result
    r = results[0]
    
    # r['rois'] bounding box

    if parked_car_boxes is None:
        # The first frame of video - assume all the cars detected are in parking spaces.
        # Save the location of each car as a parking space box and go to the next frame of video.
        parked_car_boxes = get_car_boxes(r['rois'], r['class_ids'])
        total_car_boxes = len(parked_car_boxes)
        print(f"Total parking spaces are {total_car_boxes}.")
        # get index and value of each detected space and show it
        for i, each_box in enumerate(parked_car_boxes):
            print(f"Parking space {i + 1} coordinates: ", each_box)
            # turn each box into a string and remove space
            str_park = str(each_box).replace(' ', '')
            # add each string to simple list instead of complex numpy array
            parking_list.append(str_park)    
    else:
        # We already know where the parking spaces are. Check if any are currently unoccupied.
        # Get where cars are currently located in the frame
        car_boxes = get_car_boxes(r['rois'], r['class_ids'])
        # See how much those cars overlap with the known parking spaces
        overlaps = mrcnn.utils.compute_overlaps(parked_car_boxes, car_boxes)
        
        # Loop through each known parking space box
        for parking_area, overlap_areas in zip(parked_car_boxes, overlaps):
            # For this parking space, find the max amount it was covered by any
            # car that was detected in our image
            max_IoU_overlap = np.max(overlap_areas)
            # Get the top-left and bottom-right coordinates of the parking area
            y1, x1, y2, x2 = parking_area

            # Check if the parking space is occupied by seeing if any car overlaps
            # it by more than 0.15 using IoU
            if max_IoU_overlap < 0.15:
                # Parking space not occupied! Draw a green box around it
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # Flag that we have seen at least one open space
                free_space = True
                # turn current free space to string without spaces
                free_park = str(parking_area).replace(' ', '')    
            else:
                # Parking space is still occupied - draw a red box around it
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            # Write the IoU measurement inside the box
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{max_IoU_overlap:0.2}", (x1 + 6, y2 - 6), font, 0.5, (255, 255, 255))

        # If parking space was free, start counting frames
        if free_space:
            free_space_frames += 1
        else:
            # If no spots are free, reset the count
            free_space_frames = 0

        # If a space has been free for several frames, then it is really free
        # Also we should make sure that we are seeing new free space, not a previous one 
        if free_space_frames > 2 and free_park not in free_space_list:
            # if it is a new free spot, add it to our list of free spaces
            free_space_list.append(free_park)
            free_parking_space = len(free_space_list)
        print(f"Available parking spaces: {free_parking_space}")

        # Show the frame of video on the screen
        cv2.imshow('Video', frame)

    # Hit 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up everything when finished
video_capture.release()
cv2.destroyAllWindows()
