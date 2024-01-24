import classes.camera.camera as cam
import classes.animator.animator as plotter
import matplotlib.pyplot as plt
import numpy as np
import time


if __name__ == "__main__":
    camera = cam.Camera()
    camera.streamRgb()