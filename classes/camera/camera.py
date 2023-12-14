# Camera class with object containing functions to use depthai cameras
# the idea is for it to act as a wrapper on the depthai library

import numpy as np
import depthai as dai 
import cv2
import time
import os

class Camera():
    def __init__(self):
        self.rgbResolution = dai.ColorCameraProperties.SensorResolution.THE_1080_P
        self.monoResolution = dai.MonoCameraProperties.SensorResolution.THE_480_P
        self.fps = 30 # RGB camera cannot run at 60 fps for oak d lite
        self.nn = None
        self.ret = None
        self.curr_rgb_frame = None
        self.curr_mono_frame = None
        self.curr_disparity_frame = None
    
    def streamRgb(self, calibrate = False):
        # Create pipeline
        pipeline = dai.Pipeline()

        # Create Color Camera node and set RES and FPS
        cam = pipeline.createColorCamera()
        cam.setResolution(self.rgbResolution)
        cam.setVideoSize(1920,1080)
        cam.setFps(float(self.fps))
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        # Set link with pc
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)

        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("RGB")
        cam.video.link(xout_rgb.input)

        fps = False
        display_res = ((int(cam.getVideoWidth()/2), int(cam.getVideoHeight()/2)))
        with dai.Device(pipeline) as device:
            while True:
                t1 = time.monotonic()
                queueName = device.getQueueEvent("RGB")
                t = time.monotonic() - t1
                message = device.getOutputQueue(queueName, maxSize=1, blocking = False).get()
                self.curr_rgb_frame = message.getCvFrame()
                if t > 0 and fps:
                    cv2.putText(self.curr_rgb_frame, "FPS: {0}".format(1/t), (100, 100), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 0), 1, cv2.LINE_AA, False) 
                
                im_res = cv2.resize(self.curr_rgb_frame.copy(), display_res)
                cv2.imshow("img", im_res)
                           
                # Use interactive keys from opencv
                val = cv2.waitKey(1)
                # Close windows and stop streaming
                if val in [ord('q'), ord('Q')]:
                    cv2.destroyAllWindows()
                    device.close()
                    break
                
                # If in calibration mode store images in the image buffer
                elif val == ord(' ') and calibrate:
                    self.calibration_buffer.append(self.curr_rgb_frame.copy())

                # In any mode, toggle printing fps on the image
                elif val in [ord('f'), ord('F')]:
                    fps = not(fps)
        return
    
    def streamMono(self):
        # Create pipeline
        pipeline = dai.Pipeline()

        # Create Color Camera node and set RES and FPS
        mono = pipeline.createMonoCamera()
        mono.setResolution(self.monoResolution)
        mono.setFps(float(self.fps))

        # Set link with pc
        mono.setBoardSocket(dai.CameraBoardSocket.LEFT)

        xout_mono = pipeline.createXLinkOut()
        xout_mono.setStreamName("mono")
        mono.out.link(xout_mono.input)

        fps = False

        with dai.Device(pipeline) as device:
            while True:
                t1 = time.monotonic()
                queueName = device.getQueueEvent("mono")
                t = time.monotonic() - t1
                message = device.getOutputQueue(queueName, maxSize=1, blocking = False).get()
                self.curr_mono_frame = message.getCvFrame()
                if t > 0 and fps:
                    cv2.putText(self.curr_mono_frame, "FPS: {0}".format(int(1/t)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 0), 1, cv2.LINE_AA, False) 
                cv2.imshow("img", self.curr_mono_frame)
                val = cv2.waitKey(1)
                if val in [ord('q'), ord('Q')]:
                    cv2.destroyAllWindows()
                    device.close()
                    break
                elif val in [ord('f'), ord('F')]:
                    fps = not(fps)
        return

    def streamDisparity(self):
        # Create pipeline
        pipeline = dai.Pipeline()

        # Define sources and outputs
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        depth = pipeline.create(dai.node.StereoDepth)
        xout = pipeline.create(dai.node.XLinkOut)

        xout.setStreamName("disparity")

        # Properties
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setCamera("left")
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setCamera("right")

        # Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
        depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
        depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_3x3)
        config = depth.initialConfig.get()
        config.postProcessing.spatialFilter.enable = True
        config.postProcessing.spatialFilter.holeFillingRadius = 18
        config.postProcessing.spatialFilter.alpha = 0.15
        config.postProcessing.spatialFilter.numIterations = 1
        depth.initialConfig.set(config)

        # Linking
        monoLeft.out.link(depth.left)
        monoRight.out.link(depth.right)
        depth.disparity.link(xout.input)

        # Connect to device and start pipeline
        with dai.Device(pipeline) as device:

            # Output queue will be used to get the disparity frames from the outputs defined above
            q = device.getOutputQueue(name="disparity", maxSize=1, blocking=False)

            while True:
                inDisparity = q.get()  # blocking call, will wait until a new data has arrived
                self.curr_disparity_frame = inDisparity.getFrame()
                # Normalization for better visualization
                frame = (frame * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)

                cv2.imshow("disparity", frame)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    device.close()
                    break
        return
    
    
    def calibrateOakRgb(self): # Intended to be called when for sure calibration is wanted
        cal_path = os.path.dirname(os.path.realpath(__file__))              # Get subdirectory for the camera class to store params there
        self.calibration_buffer = []

        self.streamRgb(True)       # Stream for calibration

        self._runCalibration(cal_path)
        
        for img in self.calibration_buffer:
            cv2.imshow("img", img)
            cv2.waitKey(0)
        
        cv2.destroyAllWindows()
        return
    
    def _runCalibration(self, cal_path):
        # Create charuco board and save it as an image to display
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        board = cv2.aruco.CharucoBoard((7,5),.025,.0125,dictionary)
        board_img = board.generateImage((1920, 1080))
        cv2.imwrite(cal_path+'/board.png', board_img)

        return