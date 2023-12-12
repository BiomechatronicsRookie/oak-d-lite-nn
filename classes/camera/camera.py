# Camera class with object containing functions to use depthai cameras
# the idea is for it to act as a wrapper on the depthai library

import numpy as np
import depthai as dai 
import cv2
import time

class Camera():
    def __init__(self):
        self.rgbResolution = dai.ColorCameraProperties.SensorResolution.THE_1080_P
        self.monoResolution = dai.MonoCameraProperties.SensorResolution.THE_480_P
        self.fps = 60 # RGB camera cannot run at 60 fps for oak d lite
        self.nn = None
        self.ret = None
    
    def stream_rgb(self):
        # Create pipeline
        pipeline = dai.Pipeline()

        # Create Color Camera node and set RES and FPS
        cam = pipeline.createColorCamera()
        cam.setResolution(self.rgbResolution)
        cam.setVideoSize(960,540)
        cam.setFps(float(self.fps))
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        # Set link with pc
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)

        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("RGB")
        cam.video.link(xout_rgb.input)

        with dai.Device(pipeline) as device:
            while True:
                t1 = time.monotonic()
                queueName = device.getQueueEvent("RGB")
                t = time.monotonic() - t1
                message = device.getOutputQueue(queueName, maxSize=1, blocking = False).get()
                imOut = message.getCvFrame()
                if t > 0:
                    cv2.putText(imOut, "FPS: {0}".format(1/t), (100, 100), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 0), 1, cv2.LINE_AA, False) 
                cv2.imshow("img", imOut)
                if cv2.waitKey(1) in [ord('q'), ord('Q')]:
                    cv2.destroyAllWindows()
                    device.close()
                    break
        return
    
    def stream_mono(self):
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

        with dai.Device(pipeline) as device:
            while True:
                t1 = time.monotonic()
                queueName = device.getQueueEvent("mono")
                t = time.monotonic() - t1
                message = device.getOutputQueue(queueName, maxSize=1, blocking = False).get()
                imOut = message.getCvFrame()
                if t > 0:
                    cv2.putText(imOut, "FPS: {0}".format(int(1/t)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 0), 1, cv2.LINE_AA, False) 
                cv2.imshow("img", imOut)
                if cv2.waitKey(1) in [ord('q'), ord('Q')]:
                    cv2.destroyAllWindows()
                    device.close()
                    break
        return

    def stream_disparity(self):
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
                frame = inDisparity.getFrame()
                # Normalization for better visualization
                frame = (frame * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)

                cv2.imshow("disparity", frame)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    device.close()
                    break
        return