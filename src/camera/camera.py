# Camera class with object containing functions to use depthai cameras
# the idea is for it to act as a wrapper on the depthai library

import numpy as np
import depthai as dai 
import matplotlib.pyplot as plt
import cv2
import time
import os
import warnings

## TODO: Add config file for automatic setting of stream features 
class Camera():
    def __init__(self):
        self.rgbResolution = dai.ColorCameraProperties.SensorResolution.THE_1080_P
        self.monoResolution = dai.MonoCameraProperties.SensorResolution.THE_480_P
        self.rgb_fps = 30 # RGB camera cannot run at 60 fps for oak d lite
        self.mono_fps = 60 # RGB camera cannot run at 60 fps for oak d lite
        self.nn = None
        self.ret = None
        self.curr_rgb_frame = None
        self.curr_mono_frame = None
        self.curr_disparity_frame = None
        self.mtx = None
        self.dst = None

        self._create_pipeline()
        self.device = dai.Device(self.pipeline)


    def _create_pipeline(self):
        self.pipeline = dai.Pipeline()

        # Create RGB camera and config
        self.rbg_cam = self.pipeline.createColorCamera()
        self.rbg_cam.setResolution(self.rgbResolution)
        self.rbg_cam.setVideoSize(1920,1080)
        self.rbg_cam.setFps(float(self.rgb_fps))
        self.rbg_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        self.rbg_cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
            # Create XLinkOut
        XOutRgb = self.pipeline.createXLinkOut()
        XOutRgb.setStreamName("rgb")
        self.rbg_cam.video.link(XOutRgb.input)

        # Create Mono camera and config
        self.mono = self.pipeline.createMonoCamera()
        self.mono.setResolution(self.monoResolution)
        self.mono.setFps(float(self.mono_fps))
        self.mono.setBoardSocket(dai.CameraBoardSocket.CAM_B)
            # Create XLinkOut
        XOutMono = self.pipeline.createXLinkOut()
        XOutMono.setStreamName("mono")
        self.mono.out.link(XOutMono.input)


        # Create Mono camera right and config
        self.mono_right = self.pipeline.createMonoCamera()
        self.mono_right.setResolution(self.monoResolution)
        self.mono_right.setFps(float(self.mono_fps))
        self.mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
            # Create XLinkOut
        XOutMono_right = self.pipeline.createXLinkOut()
        XOutMono_right.setStreamName("mono_right")
        self.mono_right.out.link(XOutMono_right.input)

    def stream(self, mode ='rgb'):
        if mode == 'rgb':
            display_res = ((int(self.rbg_cam.getVideoWidth()/2), int(self.rbg_cam.getVideoHeight()/2)))
        elif mode == 'mono':
            display_res = ((int(self.mono.getResolutionWidth()), int(self.mono.getResolutionHeight())))
        elif mode == 'mono_right':
            display_res = ((int(self.mono_right.getResolutionWidth()), int(self.mono_right.getResolutionHeight())))

        while True:
            queueName = self.device.getQueueEvent("{0}".format(mode))
            message = self.device.getOutputQueue(queueName, maxSize=1, blocking = False).get()
            self.curr_frame = message.getCvFrame()
            im_res = cv2.resize(self.curr_frame.copy(), display_res)
            cv2.imshow("img", im_res)
            # Use interactive keys from opencv
            val = cv2.waitKey(1)
            # Close windows and stop streaming
            if val in [ord('q'), ord('Q')]:
                cv2.destroyAllWindows()
                self.device.close()
                break
        return
    
   
    def streamWithBoardPose(self, mode = 'rgb'):
        
        calibData = self.device.readCalibration()

        if mode == 'rgb':
            display_res = ((int(self.rbg_cam.getVideoWidth()/2), int(self.rbg_cam.getVideoHeight()/2)))
            self.mtx = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 1920, 1080)).squeeze()
            self.dst = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))
        elif mode == 'mono':
            display_res = ((int(self.mono.getResolutionWidth()/2), int(self.mono.getResolutionHeight())))
            self.mtx = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, 640, 480)).squeeze()
            self.dst = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B))
        elif mode == 'mono_right':
            display_res = ((int(self.mono_right.getResolutionWidth()), int(self.mono_right.getResolutionHeight())))
            self.mtx = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, 640, 480)).squeeze()
            self.dst = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C))
        
        # Define board parameters
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        board = cv2.aruco.CharucoBoard((7,5),.025,.0125,dictionary)

        while True:
            queueName = self.device.getQueueEvent("{0}".format(mode))
            message = self.device.getOutputQueue(queueName, maxSize=1, blocking = False).get()
            self.curr_frame = message.getCvFrame()
            rot, tr = self.EstimateBoardPose(self.curr_frame, board, self.mtx, self.dst, dictionary)
            
            cv2.drawFrameAxes(self.curr_frame, self.mtx, self.dst, rot, tr, 0.1)
            im_res = cv2.resize(self.curr_frame.copy(), display_res)
            cv2.imshow("img", im_res)
                        
            # Use interactive keys from opencv
            val = cv2.waitKey(1)
            # Close windows and stop streaming
            if val in [ord('q'), ord('Q')]:
                cv2.destroyAllWindows()
                self.device.close()
                break
        return
        
    def stream2Mono(self):
        while True:
            left_queue = self.device.getQueueEvent("mono")
            right_queue = self.device.getQueueEvent("mono_right")
            message_l = self.device.getOutputQueue(left_queue, maxSize=1, blocking = False).get()
            message_r = self.device.getOutputQueue(right_queue, maxSize=1, blocking = False).get()
            l_frame = message_l.getCvFrame()
            r_frame = message_r.getCvFrame()
            self.curr_mono_frame = (l_frame/2 + r_frame/2)/255

            cv2.imshow("img", self.curr_mono_frame)
            val = cv2.waitKey(1)
            if val in [ord('q'), ord('Q')]:
                cv2.destroyAllWindows()
                self.device.close()
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
        depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
        config = depth.initialConfig.get()
        config.postProcessing.spatialFilter.enable = False
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
                self.curr_disparity_frame  = (self.curr_disparity_frame  * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)

                cv2.imshow("disparity", self.curr_disparity_frame )
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    device.close()
                    break
        return
       
    ''' CALIBRATION FUNCTION HELPERS FOR THE OAK D CAMERA'''
    
    def _runCalibration(self, cal_path):
        # Create charuco board and save it as an image to display
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        board = cv2.aruco.CharucoBoard((7,5),.025,.0125,dictionary)
        board_img = board.generateImage((1920, 1080))

        if not os.path.isfile(cal_path+'/board.png'):
            cv2.imwrite(cal_path+'/board.png', board_img)

        allCorners, allIds, imsize = self._getCalibrationKeypoitns(self.calibration_buffer, board)
        cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                    [    0., 1000., imsize[1]/2.],
                                    [    0.,    0.,           1.]])

        distCoeffsInit = np.zeros((5,1))
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)

        (ret, camera_matrix, distortion_coefficients,
        rotation_vectors, translation_vectors, _, _, errors) = cv2.aruco.calibrateCameraCharucoExtended(
                        charucoCorners=allCorners,
                        charucoIds=allIds,
                        board=board,
                        imageSize=imsize,
                        cameraMatrix=cameraMatrixInit,
                        distCoeffs=distCoeffsInit,
                        flags=flags,
                        criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
                
        fig, ax = plt.subplots(figsize = (5, 3) )
        ax.scatter(np.linspace(1, len(errors) + 1, len(errors)),errors)
        ax.plot(np.linspace(1, len(errors) + 1, len(errors)),errors)
        ax.set_xlabel('Image n')
        ax.set_ylabel('Error (pixels)')
        fig.tight_layout()
        plt.show()
        
        return camera_matrix, distortion_coefficients
    
    def _getCalibrationKeypoitns(self, files, board):

        """
        Gets the desired points of interests for each board in frame
        path: path to calbration video
        board: board with real parameters
        ---
        Corners: Corners detected by the charuco detector
        Ids: Ids detected by the charuco detector
        imsize: image shape after reshaping
        """
        allCorners = []
        allIds = []
        decimator = 0
        off = 0
        # DETECTOR INSTANTIATION FOR CHARUCO
        charuco_detector = cv2.aruco.CharucoDetector(board)
        for idx, im in enumerate(files):
            print("=> Processing image {0}".format(idx - off))
            frame = im
            if frame.ndim > 2:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
            else:
                gray = frame             
            res2 = charuco_detector.detectBoard(gray)

            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                allCorners.append(res2[0])
                allIds.append(res2[1])

            decimator+=1
            imsize = gray.shape

        return allCorners, allIds, imsize
    
    def EstimateBoardPose(self, img, board, m, d, dict = None):
        """
        Estimates the rotation and translation of the origin of the frame of the board
        mtx: camera coefficient matrix to account for projections
        dist: camera distortion coefficint matrix to account for lens distortions
        img: image with board on frame 
        board: board object with the real parameters of the board used
        dict (optional):aruco dictionary used during board generation
        ---
        r: rotation from camera space to board space
        t: translation from camera space to board space
        """
        # Change if no reprojection error has to be computed
        val = True
        # Detect board
        charuco_detector = cv2.aruco.CharucoDetector(board)
        #aruco_detector = cv2.aruco.ArucoDetector(dict)
        if img.ndim > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        res = charuco_detector.detectBoard(gray)

        # Detect pose and return rotation matrix and translation
        _, r, t = cv2.aruco.estimatePoseCharucoBoard(
                    charucoCorners = res[0],
                    charucoIds = res[1],
                    board = board,
                    cameraMatrix = m,
                    distCoeffs = d,
                    rvec = np.eye(3),
                    tvec = np.zeros(3),
                    useExtrinsicGuess = False)
    
        return r, t
    
if __name__ == '__main__':
        oak = Camera()
        oak.streamWithBoardPose('mono_right')