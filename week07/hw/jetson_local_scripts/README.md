# Tensorflow Face Detector
A mobilenet SSD(single shot multibox detector) based face detector with pretrained model provided, powered by tensorflow [object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection), trained by [WIDERFACE dataset](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/).

## Dependencies
- Tensorflow > 1.2
- OpenCV python

## Usage

The code included runs on a Jetson TX2 edge computer 

### Prepare pre-trained model
Click [here](https://drive.google.com/open?id=0B5ttP5kO_loUdWZWZVVrN2VmWFk) to download the pre-trained model from google drive.
Put the model under the model folder.

### Connect a USB camera to Jetson TX2

### Run video detection
At the source root
```bash
python3 inference_usbCam_face.py cameraID=<0|1>
```
The docker-compose is included to deploy the whole application (include MQTT messaging broker and clients) with a single command.
```bash 
docker-compose up -d
```
The script crops the faces from the streaming video and sends them to a IBM cloud broker via the MQTT forwarding client started on Jetson TX2 above. The binary files are stored in an object storage (The scripts run on the IBM cloud instance are also included). 

## License
Usage of the code and model by yeephycho is under the license of Apache 2.0.

The code is based on GOOGLE tensorflow object detection api. Please refer to the license of tensorflow.

Dataset is based on WIDERFACE dataset. Please refer to the license to the WIDERFACE license.
