# Camera class with object containing functions to use depthai cameras
# the idea is for it to act as a wrapper on the depthai library

import numpy as np
import depthai as dai 
import matplotlib.pyplot as plt
import cv2
import time
import os
import warnings

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
    
    def streamRgb(self, calibrate = False):
        # Create pipeline
        pipeline = dai.Pipeline()

        # Create Color Camera node and set RES and FPS
        cam = pipeline.createColorCamera()
        cam.setResolution(self.rgbResolution)
        cam.setVideoSize(1920,1080)
        cam.setFps(float(self.rgb_fps))
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        # Set link with pc
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("RGB")
        cam.video.link(xout_rgb.input)

        fps = False
        self.recording_rgb = False
        display_res = ((int(cam.getVideoWidth()/2), int(cam.getVideoHeight()/2)))

        with dai.Device(pipeline) as device:
            while True:
                t1 = time.monotonic()
                queueName = device.getQueueEvent("RGB")
                t = time.monotonic() - t1
                message = device.getOutputQueue(queueName, maxSize=1, blocking = False).get()
                self.curr_rgb_frame = message.getCvFrame()
                if self.recording_rgb:
                    output.write(self.curr_rgb_frame)
                if t > 0 and fps:
                    cv2.putText(self.curr_rgb_frame, "FPS: {0}".format(1/t), (100, 100), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 0), 1, cv2.LINE_AA, False) 
                im_res = cv2.resize(self.curr_rgb_frame.copy(), display_res)
                cv2.imshow("img", im_res)
                           
                # Use interactive keys from opencv
                val = cv2.waitKey(1)
                # Close windows and stop streaming
                if val in [ord('q'), ord('Q')]:
                    if self.recording_rgb:
                        self.recording_rgb = False
                        output.release()
                    cv2.destroyAllWindows()
                    device.close()
                    break
                
                # If in calibration mode store images in the image buffer
                elif val == ord(' ') and calibrate:
                    self.calibration_buffer.append(self.curr_rgb_frame.copy())
                    print('Img {0} of 15'.format(len(self.calibration_buffer)))
                    if len(self.calibration_buffer) == 15:
                        cv2.destroyAllWindows()
                        device.close()
                        break

                # In any mode, toggle printing fps on the image
                elif val in [ord('f'), ord('F')]:
                    fps = not(fps)
                
                elif val in [ord('r'), ord('R')]:
                    if self.recording_rgb:
                        self.recording_rgb = False
                        output.release() 
                    
                    else:
                        if self.recording_rgb == False:
                            video_dir = os.getcwd() + '\\data\\videos\\'
                            video_name = '{0}.mp4'.format(str(len(os.listdir(video_dir))).zfill(4))
                            output = cv2.VideoWriter(video_dir + video_name, cv2.VideoWriter_fourcc(*'mp4v'), self.rgb_fps, cam.getVideoSize())
                            self.recording_rgb = True  
        return

    def streamRgbWithBoardPose(self):
        # Create pipeline
        pipeline = dai.Pipeline()

        # Create Color Camera node and set RES and FPS
        cam = pipeline.createColorCamera()
        cam.setResolution(self.rgbResolution)
        cam.setVideoSize(1920,1080)
        cam.setFps(float(self.rgb_fps))
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        # Set link with pc
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("RGB")
        cam.video.link(xout_rgb.input)

        fps = False
        display_res = ((int(cam.getVideoWidth()/2), int(cam.getVideoHeight()/2)))

        with dai.Device(pipeline) as device:
            calibData = device.readCalibration()
            self.mtx = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 1920, 1080)).squeeze()
            self.dst = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))
            # Define board parameters
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
            board = cv2.aruco.CharucoBoard((7,5),.025,.0125,dictionary)

            while True:
                t1 = time.monotonic()
                queueName = device.getQueueEvent("RGB")
                t = time.monotonic() - t1
                message = device.getOutputQueue(queueName, maxSize=1, blocking = False).get()
                self.curr_rgb_frame = message.getCvFrame()
                rot, tr = self.EstimateBoardPose(self.curr_rgb_frame, board, dictionary)

                if t > 0 and fps:
                    cv2.putText(self.curr_rgb_frame, "FPS: {0}".format(1/t), (100, 100), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 1, cv2.LINE_AA, False)
                cv2.drawFrameAxes(self.curr_rgb_frame, self.mtx, self.dst, rot, tr, 0.1)
                im_res = cv2.resize(self.curr_rgb_frame.copy(), display_res)
                cv2.imshow("img", im_res)
                           
                # Use interactive keys from opencv
                val = cv2.waitKey(1)
                # Close windows and stop streaming
                if val in [ord('q'), ord('Q')]:
                    cv2.destroyAllWindows()
                    device.close()
                    break

                # In any mode, toggle printing fps on the image
                elif val in [ord('f'), ord('F')]:
                    fps = not(fps)
        return
    
    
    def streamMono(self, side = None):
        # Create pipeline
        pipeline = dai.Pipeline()

        # Create Color Camera node and set RES and FPS
        mono = pipeline.createMonoCamera()
        mono.setResolution(self.monoResolution)
        mono.setFps(float(self.mono_fps))

        # Set link with pc
        if side == 'LEFT' or side == None:
            mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
            if not side:
                warnings.warn("No camera was selected. Using mono LEFT camera...")
        if side == 'RIGHT':
            mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)

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
    
    def stream2Mono(self, calibrate = False):
        # Create pipeline
        pipeline = dai.Pipeline()

        # Create Color Camera node and set RES and FPS
        mono_r = pipeline.createMonoCamera()
        mono_r.setResolution(self.monoResolution)
        mono_r.setFps(float(self.mono_fps))
        mono_l = pipeline.createMonoCamera()
        mono_l.setResolution(self.monoResolution)
        mono_l.setFps(float(self.mono_fps))

        # Set link with pc
        mono_r.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        mono_l.setBoardSocket(dai.CameraBoardSocket.LEFT)

        xout_mono_r = pipeline.createXLinkOut()
        xout_mono_r.setStreamName("mono_r")
        xout_mono_l = pipeline.createXLinkOut()
        xout_mono_l.setStreamName("mono_l")
        mono_r.out.link(xout_mono_r.input)
        mono_l.out.link(xout_mono_l.input)

        fps = False

        with dai.Device(pipeline) as device:
            while True:
                t1 = time.monotonic()
                left_queue = device.getQueueEvent("mono_l")
                right_queue = device.getQueueEvent("mono_r")
                t = time.monotonic() - t1
                message_l = device.getOutputQueue(left_queue, maxSize=1, blocking = False).get()
                message_r = device.getOutputQueue(right_queue, maxSize=1, blocking = False).get()
                l_frame = message_l.getCvFrame()
                r_frame = message_r.getCvFrame()
                self.curr_mono_frame = (l_frame/2 + r_frame/2)/255

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

                elif val == ord(' ') and calibrate:
                    self.calibration_buffer_left.append(l_frame.copy())
                    self.calibration_buffer_right.append(r_frame.copy())
                    print('Imgs {0} of 15'.format(len(self.calibration_buffer_left)))
                    if len(self.calibration_buffer_left) == 15:
                        cv2.destroyAllWindows()
                        device.close()
                        break
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
    
    
    def calibrateOakRgb(self): # Intended to be called when for sure calibration is wanted
        cal_path = os.path.dirname(os.path.realpath(__file__))              # Get subdirectory for the camera class to store params there
        self.calibration_buffer = []

        self.streamRgb(True)       # Stream for calibration

        mtx, dst = self._runCalibration(cal_path)
        # Override calibration parameters inside the camera:
        pipeline = dai.Pipeline()

        # Create Color Camera node and set RES and FPS
        cam = pipeline.createColorCamera()
        with dai.Device(pipeline) as device:
            calibData = device.readCalibration()
            calibData.setCameraIntrinsics(dai.CameraBoardSocket.CAM_A, mtx.tolist(), 1920, 1080)
            calibData.setDistortionCoefficients(dai.CameraBoardSocket.CAM_A, dst.ravel().tolist())
            device.close()

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        board = cv2.aruco.CharucoBoard((7,5),.025,.0125,dictionary)
        
        for img in self.calibration_buffer:
            r, t = self.EstimateBoardPose(img, board, self.mtx, self.dst, dictionary)
            cv2.drawFrameAxes(img, self.mtx, self.dst, r, t, 0.1)
            cv2.imshow("img", img)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

        return
    
    def calibrateOak2Mono(self): # Intended to be called when for sure calibration is wanted
        cal_path = os.path.dirname(os.path.realpath(__file__))              # Get subdirectory for the camera class to store params there
        self.calibration_buffer_left = []
        self.calibration_buffer_right = []
        buff = [self.calibration_buffer_left, self.calibration_buffer_right]
        sockets = [dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C]        # Left, Right

        self.stream2Mono(True)       # Stream for calibration

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        board = cv2.aruco.CharucoBoard((7,5),.025,.0125,dictionary)

        for i in range(0,2):
            self.calibration_buffer = buff[i]
            mtx, dst = self._runCalibration(cal_path)
            # Override calibration parameters inside the camera:
            pipeline = dai.Pipeline()

            with dai.Device(pipeline) as device:
                calibData = device.readCalibration()
                calibData.setCameraIntrinsics(sockets[i], mtx.tolist(), 640, 480)
                calibData.setDistortionCoefficients(sockets[i], dst.ravel().tolist())
                device.close()

            for img in self.calibration_buffer:
                r, t = self.EstimateBoardPose(img, board, mtx, dst, dictionary)
                cv2.drawFrameAxes(img, mtx, dst, r, t, 0.1)
                cv2.imshow("img", img)
                cv2.waitKey(0)

        cv2.destroyAllWindows()
        
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