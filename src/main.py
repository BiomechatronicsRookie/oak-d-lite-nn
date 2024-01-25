import camera.camera as cam
import animator.animator as plotter
import matplotlib.pyplot as plt
import numpy as np
import time


if __name__ == "__main__":
    camera = cam.Camera()
    camera.stream()
    camera.streamWithBoardPose()

    if not camera.device.isClosed():
        camera.device.close()