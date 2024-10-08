# Crowd Analysis


### Usage

Run the script with the following command:

```
1) Download below tar file and extract them under the testdata directory
Link: https://drive.google.com/file/d/1CBJemG88zV2_I9bL_UrgE-5Fmy4P4ElC/view?usp=sharing

2) python3 count.py --gender_model_path ./testdata/effb0_best.pth --detector_model_path ./testdata/yolov8m.pt --source_path ./testdata/test.mp4 --line_y_position 250
```


This will start processing the video, detecting objects, predicting their gender, and displaying the annotated frames.


## Install:

### pip :
Need a Python>=3.8, 
```
pip3 install -r requirements.txt
```

### conda :
```
conda create -n count python=3.8
pip3 install -r requirements.txt

```


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




