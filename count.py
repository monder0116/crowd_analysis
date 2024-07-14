import cv2
import supervision as sv
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultralytics import YOLO
import torch
from torchvision import models, transforms
import torch.nn as nn
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# EfficientNetB0 Singleton Class
class EfficientNetB0Singleton:
    _instance = None
    _genderstr = ["Man", "Woman"]

    def __new__(cls, model_path):
        if cls._instance is None:
            cls._instance = super(EfficientNetB0Singleton, cls).__new__(cls)
            model = models.efficientnet_b0(pretrained=True)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, 2)  # 2-class output layer
            map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.load_state_dict(torch.load(model_path, map_location=map_location), strict=False)
            cls._instance.model = model.to(device)
            cls._instance.model.eval()
            cls._instance.preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return cls._instance

    def predict(self, cropped_obj):
        input_tensor = self.preprocess(cropped_obj).unsqueeze(0).to(device)
        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted = torch.max(output, 1)
        return self._genderstr[predicted.item()]

def process_gender_detection(i, detections, frame, crossed_in, crossed_out, efficient_net_model):
    trackid = detections.tracker_id[i]
    xyxy = detections.xyxy[i]
    label = ""
    found = False
    if crossed_in[i] or crossed_out[i]:
        x1, y1, x2, y2 = xyxy
        cropped_obj = frame[int(y1):int(y2), int(x1):int(x2)]
        gender_label = efficient_net_model.predict(cropped_obj)
        label = f"Gender: {gender_label} #{trackid}"
        found = True
    else:
        label = f"#{trackid}"
    return {"index": i, "label": label, "found": found}

class AnnotationManager:
    def __init__(self, line_y_position, video_width, video_height):
        self.line_y_position,self.video_width,self.video_height=line_y_position,video_width,video_height
        self.tracker = sv.ByteTrack()
        self.trace_annotator = sv.TraceAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.box_annotator = sv.BoxCornerAnnotator()
        start, end = sv.Point(x=0, y=line_y_position), sv.Point(x=video_width, y=line_y_position)
        self.line_zone = sv.LineZone(start=start, end=end, triggering_anchors=[sv.Position.CENTER])
        self.line_zone_annotator = sv.LineZoneAnnotator(
            thickness=4,
            text_thickness=4,
            text_scale=2,
            display_out_count=False,
            display_in_count=False
        )
        self.PASSED_IDS = {}

    def annotate_frame(self, frame, detections, crossed_in, crossed_out, results):
        labels = []
        for i in range(len(detections.tracker_id)):
            res = results[i]
            found = res['found']
            if found:
                self.PASSED_IDS[detections.tracker_id[i]] = res['label']
                labels.append(res['label'])
            else:
                trackid = detections.tracker_id[i]
                if trackid in self.PASSED_IDS.keys():
                    labels.append(self.PASSED_IDS[trackid])
                else:
                    labels.append(res['label'])

        annotated_frame = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.line_zone_annotator.annotate(annotated_frame, line_counter=self.line_zone)
        crossed_count_in = self.line_zone.in_count
        crossed_count_out = self.line_zone.out_count
        cv2.putText(annotated_frame, f"Crossed In: {crossed_count_in}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Crossed Out: {crossed_count_out}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return annotated_frame

def main(model_path, yolov8_model_path, source_path, conf_thres, iou_thres, line_y_position):
    efficient_net_model = EfficientNetB0Singleton(model_path=model_path)
    model = YOLO(yolov8_model_path,verbose=False)
    video_info = sv.VideoInfo.from_video_path(video_path=source_path)
    video_width = int(video_info.width)
    video_height = int(video_info.height)
    del video_info
    annotation_manager = AnnotationManager(line_y_position=line_y_position,video_width=video_width, video_height=video_height)
    frames_generator = sv.get_video_frames_generator(source_path=source_path)
    for frame in frames_generator:

        results = model(frame, conf=conf_thres,iou=iou_thres,verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        filter_by_class = detections.class_id == 0
        detections = annotation_manager.tracker.update_with_detections(detections[filter_by_class])
        crossed_in, crossed_out = annotation_manager.line_zone.trigger(detections)
        results = {}

        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_index = {executor.submit(process_gender_detection, i, detections, frame, crossed_in, crossed_out, efficient_net_model): i for i in range(len(detections.tracker_id))}
            for future in as_completed(future_to_index):
                index_value = future.result()
                index = index_value['index']
                value = index_value['label']
                found = index_value['found']
                results[index] = {"label": value, "found": found}

        annotated_frame = annotation_manager.annotate_frame(frame, detections, crossed_in, crossed_out, results)

        cv2.imshow("Annotated Frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO and EfficientNetB0 Video Processing")
    parser.add_argument("--gender_model_path", type=str, required=True, help="Path to the EfficientNet model file")
    parser.add_argument("--detector_model_path", type=str, required=True, help="Path to the YOLOv8 model file")
    parser.add_argument("--source_path", type=str, required=True, help="Path to the video file")
    parser.add_argument("--conf_thres", type=float, default=0.25, help="Confidence threshold for YOLOv8")
    parser.add_argument("--iou_thres", type=float, default=0.45, help="IoU threshold for YOLOv8")
    parser.add_argument("--line_y_position", type=int, required=True,default=400, help="Y-axis position for the line")
    args = parser.parse_args()

    main(args.gender_model_path, args.detector_model_path, args.source_path, args.conf_thres, args.iou_thres, args.line_y_position)
