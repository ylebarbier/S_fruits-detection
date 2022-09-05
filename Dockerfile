FROM cytomineuliege/software-python3-base:latest
RUN mkdir /fruits_detection
RUN mkdir /fruits_detection/yolov5
RUN mkdir /fruits_detection/exp
RUN chmod -R 777 /fruits_detection
RUN git clone https://github.com/ultralytics/yolov5.git /fruits_detection/yolov5
RUN pip install -r /fruits_detection/yolov5/requirements.txt
RUN pip install opencv-contrib-python-headless==4.5.5.62
RUN pip install opencv-python-headless==4.5.5.62
RUN pip install opencv-python==4.5.5.62
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install shapely
ADD app.py /fruits_detection/app.py
ADD yolo.pt /fruits_detection/yolo.pt
ENTRYPOINT ["python", "/fruits_detection/app.py"]
