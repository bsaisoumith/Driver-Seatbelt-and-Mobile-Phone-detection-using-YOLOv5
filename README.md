Driver Seatbelt and Mobile Phone Detection using YOLOv5

This project utilizes YOLOv5, a state-of-the-art object detection algorithm, 
to detect whether a driver is wearing a seatbelt and whether they are using a mobile phone while driving. 
The aim is to enhance driver safety by providing real-time alerts or monitoring.

## Features
- Detects seatbelt usage and mobile phone presence in the driver's seat.
- Real-time detection for on-the-fly monitoring.
- Easy integration with existing systems.

## Requirements

- Python 3.x
- PyTorch (refer to the [official PyTorch website](https://pytorch.org/) for installation instructions)
- OpenCV
  ```bash
  pip install opencv-python
- YOLOv5 (install using the instructions provided in the [YOLOv5 GitHub repository](https://github.com/ultralytics/yolov5))

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/bsaisoumith/Driver-Seatbelt-and-Mobile-Phone-detection-using-YOLOv5

2. Navigate to the project directory:
   cd Driver-Seatbelt-and-Mobile-Phone-detection-using-YOLOv5
   
4. Install the required dependencies:
   pip install -r requirements.txt

5. Download the YOLOv5 pre-trained weights:
   python download_weights.py

## Usage

To detect driver seatbelt and mobile phone usage in an image, run the following command:
python detect.py --source path/to/your/image.jpg

To detect in a video stream, run:
python detect.py --source 0

## Customization
If you want to train the model on your custom dataset, refer to the YOLOv5 documentation for training instructions.
You can use the provided data.yaml file for configuration.

## Results
Include sample images or video frames with the detections highlighted to showcase the performance of your model.
