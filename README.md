# Crowd Analysis

## Detail and Usage:
Video Processing Pipeline with YOLOv8 and EfficientNetB0

This repo processes a video to detect objects and predict their gender, then annotates the video frames.

### Key Components
* EfficientNetB0Singleton Class :

    Loads and uses the EfficientNetB0 model for gender classification.
    Prepares input images and predicts the gender ("man" or "woman").
* process_gender_detection Function : 

    Processes detected objects in the video.
    Crops each object and predicts its gender.
    Returns a label with the gender and tracking index. <b>This function is executed in parallel using a ThreadPoolExecutor for efficiency</b>
* AnnotationManager Class : 

    Manages the annotation of video frames.
    Tracks objects crossing a specified line.
    Annotates frames with bounding boxes, labels, and traces.
* main Function :

    Loads models and initializes video processing.
    Processes video frames, detects objects with YOLOv8, and tracks them. <b>Uses a ThreadPoolExecutor to run process_gender_detection in parallel for each detected object. </b >
    Annotates frames with detection results and displays them.
    Stops processing when the 'q' key is pressed.

### Usage

Run the script with the following command:

```
python count.py --gender_model_path path/to/efficientnet_model --detector_model_path path/to/yolov8_model --source_path path/to/video --conf_thres 0.25 --iou_thres 0.45 --line_y_position 400
```


This will start processing the video, detecting objects, predicting their gender, and displaying the annotated frames.




## Install:

### pip :
Need a Python>=3.8, 
```
pip3 install supervision
```

### conda :
```

conda install -c conda-forge supervision
```

