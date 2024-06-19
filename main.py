import time
import logging
import math
import queue
import threading

import cv2
import cvzone
import requests
import yaml

from ultralytics import YOLO


with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

video_path = config['video_path']

model_path = config['model']['path']
model = YOLO(model_path)

classNames = config['classNames']
class_count = ["box"]

NUM_WORKER_THREADS = 100
FRAME_SKIP_INTERVAL = 10

stop_thread = False
# Logging configuration
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),
                        logging.StreamHandler()
                    ])


def measure_runtime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Runtime of {func.__name__}: {runtime} seconds")
        return result

    return wrapper


input_frame_queue = queue.Queue(maxsize=NUM_WORKER_THREADS * 2)
processed_queue = queue.Queue(maxsize=NUM_WORKER_THREADS * 2)
detections_queue = queue.Queue(maxsize=NUM_WORKER_THREADS * 2)
detect_action_queue = queue.Queue(maxsize=NUM_WORKER_THREADS * 2)

@measure_runtime
def detect_objects(frame):
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            current_class = classNames[cls]
            if current_class in class_count:
                detections_queue.put({'class': current_class, 'conf': conf, 'coord': box})


def process_frame(frame):
    print("Start individual threads for model detection: Frames")

    detections_thread1 = threading.Thread(target=detect_objects, args=(frame,))
    detections_thread1.start()

    detections_thread1.join()

    while not detections_queue.empty():

        detection_marking = detections_queue.get()
        class_label = detection_marking["class"]
        conf = detection_marking["conf"]
        x1, y1, x2, y2 = detection_marking['coord'].xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        Length = x2 - x1
        width = y2 - y1
        Length = Length / 58.7
        Length = round(Length, 2)
        width = width / 58.7
        width = round(width, 2)
        text = f"Length: {Length} cm, width: {width} cm"
        position = (max(0, x1), max(10, y1 - 10))

        # D = round(D, 2)
        # cvzone.putTextRect(frame,  f"{D} mm", (max(0, x1), max(10, y1)),
        #                    thickness=3, offset=2)
        cvzone.putTextRect(frame, text, position, thickness=3, offset=2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 4)

        detection_marker = True

    # if detection_marker:
    #     detect_action_queue.put({'frame': frame.copy(), 'class': class_label, 'conf': conf})

    # detections_queue.queue.clear()
    print("detections_queue 3  " + str(detections_queue.qsize()))
    print("Processed individual frame.")
    return frame


def capture_frames(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    #cap = cv2.VideoCapture(0)
    frame_count = 0
    while not stop_thread:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % FRAME_SKIP_INTERVAL == 0:
            input_frame_queue.put(frame)
        frame_count += 1


def process_frames():
    while not stop_thread:
        if not input_frame_queue.empty():
            print(f'input_frame_queue size: {input_frame_queue.qsize()}')
            input_frame = input_frame_queue.get()
            processed_frame = process_frame(input_frame)
            processed_queue.put(processed_frame if processed_frame is not None else input_frame)
    print(f'Exiting method process_frames()')


def display_frames():
    global stop_thread
    while not stop_thread:
        processed_frame = processed_queue.get()
        print(f'Process Queue size{processed_queue.qsize()}')
        width = 640
        height = 480
        resized_frame = cv2.resize(processed_frame, (width, height))
        cv2.imshow('Processed Frame', resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_thread = True
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        rtsp_url = "http://192.168.1.25:8080/video"

        capture_thread = threading.Thread(target=capture_frames, args=(rtsp_url,))
        capture_thread.start()

        process_thread = threading.Thread(target=process_frames, )
        process_thread.start()

        display_thread = threading.Thread(target=display_frames, )
        display_thread.start()

        all_threads = [capture_thread, process_thread, display_thread]

        for t in all_threads:
            t.start()
            print(f'Starting Thread {t.name} ...')
        for t in all_threads:
            print(f'Awaiting Thread {t.name} ...')
            t.join()
            print(f'Completion Thread {t.name} ...')
    except Exception as e:
        print(e)
    finally:
        while not input_frame_queue.empty():
            input_frame_queue.get()
        while not detections_queue.empty():
            detections_queue.get()
        while not processed_queue.empty():
            processed_queue.get()
        print("Exiting Application")
