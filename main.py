#import classes.camera.camera as cam
import classes.animator.animator as plotter
import matplotlib.pyplot as plt
import numpy as np
import time

def main():
    x_dim = 100
    y_dim = 100
    plt.ion()
    animator = plotter.Animator3D()
    animator.initialize_canvas((x_dim, y_dim))

    while animator.state:
        t1 = time.monotonic()
        new_data = np.random.rand(x_dim, y_dim)/10
        animator.update_canvas(new_data)
        t = (time.monotonic() - t1)
        if t:
            print(1/t)
    return

def animation_2D():
    plt.ion()
    animator = plotter.Animator2D()
    animator.initialize_canvas(100)
    x, y = np.linspace(0,1,100), np.random.rand(100)
    while animator.state:
        t1 = time.monotonic()
        new_data = np.random.rand(100)
        animator.update_canvas(new_data)
        t = (time.monotonic() - t1)
        if t:
            print(1/t)
    return

if __name__ == "__main__":
    main()