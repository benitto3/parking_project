import os
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path

# Getting rid of Tensorflow information and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuration for the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes +
                          # one background class

# Filter only detected cars
def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car, motorcycle or truck, skip it
        if class_ids[i] in [3, 4, 6]:
            car_boxes.append(box)

    return np.array(car_boxes)

# Root directory of the project
ROOT_DIR = '.'

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Value 0 or 1 without "" for camera input
VIDEO_SOURCE = 'sample/video/many_out_in.mp4'

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode='inference', model_dir=MODEL_DIR, \
                                config=MaskRCNNConfig())
# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Load the video file we want to run detection on
video_capture = cv2.VideoCapture(VIDEO_SOURCE)

# Location of parking spaces
parked_car_boxes = None
# list of all parking spaces
parking_list = []
# list of all detected free parking spaces
free_space_list = []
# list for waiting to be added to free space list, if we are sure the space is free
free_space_waitlist = []
# frames of free parking space in a row
free_space_frames = 0
# total parking spaces number
total_car_boxes = 0
# total free parking spaces number
free_parking_space = 0
# check is this the first detection
first_detection = True

# Function for opening file, overwriting it and saving
def detect(filename, text):
    with open(filename, "w") as file:
        file.write(text)

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

    if parked_car_boxes is None:
        # The first frame of video - assume all the cars detected are
        # in parking spaces.
        # Save the location of each car as a parking space box 
        # and go to the next frame of video.
        # r['rois'] bounding box
        parked_car_boxes = get_car_boxes(r['rois'], r['class_ids'])
        total_car_boxes = len(parked_car_boxes)
        print(f"Барлық көлік тұрағының саны {total_car_boxes}.")
        # get index and value of each detected space and show it
        for i, each_box in enumerate(parked_car_boxes):
            # Getting rid of unusual spaces for consistency
            box_without_spaces = str(each_box).replace('[ ', '[')\
                                              .replace('  ', ' ')
            print(f"№ {i + 1} көлік тұрағы координаттары: ", box_without_spaces)
            # add each string to a simple list instead of complex numpy array
            parking_list.append(box_without_spaces)
            
        # Overwrite html so it could show us total number of parking spaces
        detect("parking_kaz.html", f"""
        <!DOCTYPE html>
        <html>
           <head>
               <title>Көлік тұрағын бақылау</title>
           </head>                    
        <body>
           <p>Жалпы көлік тұрағының саны: { total_car_boxes }</p>
           <img width="50%" src="result/first_detection.jpg" alt="Табылған жалпы көлік тұрақтары."/> <br />
           <p>Босаған көлік тұрақтарының саны: { free_parking_space }</p>
           <img width="50%" src="result/last_detection.jpg" alt="Табылған бос көлік тұрақтары."/> <br />
        </body>
        </html>""")
             
    else:
        # We already know where the parking spaces are. 
        # Check if any are currently unoccupied.
        # Get where cars are currently located in the frame
        car_boxes = get_car_boxes(r['rois'], r['class_ids'])
        # See how much those cars overlap with the known parking spaces
        overlaps = mrcnn.utils.compute_overlaps(parked_car_boxes, car_boxes)
        
        # Loop through each known parking space box
        for parking_area, overlap_areas in zip(parked_car_boxes, overlaps):
            # For this parking space, find the max amount it was covered
            # by any car that was detected in our image
            max_IoU_overlap = np.max(overlap_areas)
            # Get the top-left and bottom-right coordinates of
            # the parking area
            y1, x1, y2, x2 = parking_area

            # turn current free space to string without spaces
            current_parking = str(parking_area).replace(' ', '')
            
            # Check if the parking space is occupied by seeing if this
            # specific car overlaps it by more than 0.15 using IoU
            if max_IoU_overlap < 0.15:
                # Parking space not occupied! Draw a green box around it
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # Start counting frames to be sure the space is free
                free_space_frames += 1
                # Add this specific parking space to the waitlist until we verify it is free
                free_space_waitlist.append(current_parking)
                # The space is free and we did not store it yet
                if free_space_frames > 1 and \
                   current_parking not in free_space_list:
                    # Then add the space to the list of free parking spaces
                    free_space_list.append(current_parking)
                    
            else:
                # Parking space is still occupied - draw a red box around it
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                if current_parking in free_space_waitlist:
                    # We reset the counter ONLY if this space is in waiting
                    # list, which means it was a false detection
                    free_space_frames = 0
                # If the space is occupied now, we remove it from
                # the list of free spaces
                if current_parking in free_space_list:
                    free_space_list.remove(current_parking)
                
            # Write the IoU measurement inside the box
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"Max IoU: {max_IoU_overlap:0.2}", \
                                        (x1 + 6, y2 - 6), font, 0.5, (255, 255, 255))                   
            
        # How many free parking spaces do we have? Show it.
        free_parking_space = len(free_space_list)        
        print(f"Қол жетімді көлік тұрақтары саны: {free_parking_space}")
        
        # Save the last detection image
        cv2.imwrite('result/last_detection.jpg', frame)
        # If it is the first frame of detection, save it
        if first_detection:
            cv2.imwrite('result/first_detection.jpg', frame)
            first_detection = False

        # Overwrite html so we could see number of free parking spaces
        detect("parking_kaz.html", f"""
        <!DOCTYPE html>
        <html>
           <head>
              <title>Көлік тұрағын бақылау</title>
           </head>                                 
           <body>
              <p>Жалпы көлік тұрағының саны: { total_car_boxes }</p>
              <img width="50%" src="result/first_detection.jpg" alt="Табылған жалпы көлік тұрақтары."/> <br />
              <p>Босаған көлік тұрақтарының саны: { free_parking_space }</p>
              <img width="50%" src="result/last_detection.jpg" alt="Табылған бос көлік тұрақтары."/> <br />
           </body>
        </html>""")

        # Show the frame of video on the screen
        cv2.imshow('Video stop "Q"', frame)

    # Hit 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up everything when finished
video_capture.release()
cv2.destroyAllWindows()
