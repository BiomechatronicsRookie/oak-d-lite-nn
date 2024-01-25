import numpy as np
import time
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation as anim

class Animator2D():
    def __init__(self):
        self.fig_2d = None
        self.axs_2d = None
        self.data = None
        self.bg = None
        self.x = None
        self.y = None
        self.create_canvas()
        self.state = False

    def create_canvas(self):
        self.fig_2d, self.axs_2d = plt.subplots()
        self.fig_2d.canvas.mpl_connect('close_event', self.on_close)
        self.fig_2d.canvas.mpl_connect('draw_event', self.on_draw)
        
    def initialize_canvas(self, dims):
        dim_x = dims
        self.x = np.linspace(0, dim_x, dim_x)
        self.data, = self.axs_2d.plot(self.x, self.y, linewidth = 2, animated = True)
        self.fig_2d.add_artist(self.data)
        self.f
        self.show_canvas()
    
    def update_canvas(self, new_data):
        cv = self.fig_2d.canvas
        cv.restore_region(self.bg)
        self.data.set_data(self.x,new_data)
        cv.figure.draw_artist(self.data)
        cv.blit(self.fig_2d.canvas.figure.bbox)
        cv.flush_events()

    def show_canvas(self):
        plt.show(block = False)
        plt.pause(.1) # Just needed to add a small pause...
        self.bg = self.fig_2d.canvas.copy_from_bbox(self.fig_2d.canvas.figure.bbox)
        self.state = True
        
    def on_close(self, event): # Apparently event is necessary
        self.state = False

    def on_draw(self, event):
        cv = self.fig_2d.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self.bg = cv.copy_from_bbox(cv.figure.bbox)

class Animator3D():
    def __init__(self):
        self.fig_3d = None
        self.axs_3d = None
        self.data = None
        self.bg = None
        self.x = None
        self.y = None
        self.create_canvas()
        self.state = False

    def create_canvas(self):
        self.fig_3d, self.axs_3d = plt.subplots(subplot_kw = dict(projection='3d'))
        self.fig_3d.canvas.mpl_connect('close_event', self.on_close)
        self.fig_3d.canvas.mpl_connect('draw_event', self.on_draw)
        
    def initialize_canvas(self, data_dim):
        dim_x, dim_y = data_dim
        a = np.linspace(0, dim_x, dim_x)
        b = np.linspace(0, dim_y, dim_y)
        xv, yv = np.meshgrid(a, b)
        zv = np.zeros(data_dim)
        self.x, self.y, z = xv.ravel(), yv.ravel(), zv.ravel()
        self.data, = self.axs_3d.plot(self.x, self.y, z, linewidth = 2 ,linestyle="", marker ="o", animated = True)
        self.fig_3d.add_artist(self.data)
        self.axs_3d.set_zlim(-1, 1)
        self.show_canvas()
    
    def update_canvas(self, new_data):
        cv = self.fig_3d.canvas
        cv.restore_region(self.bg)
        self.data.set_data(self.x,self.y)
        self.data.set_3d_properties(new_data.ravel())
        cv.figure.draw_artist(self.data)
        cv.blit(self.fig_3d.canvas.figure.bbox)
        cv.flush_events()

    def show_canvas(self):
        plt.show(block = False)
        plt.pause(.1) # Just needed to add a small pause...
        self.bg = self.fig_3d.canvas.copy_from_bbox(self.fig_3d.canvas.figure.bbox)
        self.state = True
        
    def on_close(self, event): # Apparently event is necessary
        self.state = False

    def on_draw(self, event):
        cv = self.fig_3d.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self.bg = cv.copy_from_bbox(cv.figure.bbox)

def animation_2D():
    plt.ion()
    animator = Animator2D()
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

def animation_3D():
    x_dim = 100
    y_dim = 100
    plt.ion()
    animator = Animator3D()
    animator.initialize_canvas((x_dim, y_dim))

    while animator.state:
        t1 = time.monotonic()
        new_data = np.random.rand(x_dim, y_dim)/10
        animator.update_canvas(new_data)
        t = (time.monotonic() - t1)
        if t:
            print(1/t)
    return

if __name__ == '__main__':
    animation_3D()