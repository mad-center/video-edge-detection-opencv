import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np


# The first argument num will be the next value in *frames*
def update_line(num, data, line):
    line.set_data(data[..., :num])
    return line,


def example_01():
    fig1 = plt.figure()
    # Fixing random state for reproducibility
    np.random.seed(19680801)
    data = np.random.rand(2, 25)
    # red solid line
    l, = plt.plot([], [], 'r-')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('x')
    plt.title('test')
    # If ``blit == True``, *func* must return an iterable of all artists that were modified or created
    line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
                                       interval=50, blit=True)

    plt.show()

    # To save the animation, use the command:
    # line_ani.save('lines.mp4')


def example_02():
    fig2 = plt.figure()

    x = np.arange(-9, 10)
    y = np.arange(-9, 10).reshape(-1, 1)
    # hypot: Equivalent to sqrt(x1**2 + x2**2)
    base = np.hypot(x, y)
    ims = []
    for add in np.arange(15):
        ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),))

    im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
                                       blit=True)
    # To save this second animation with some metadata, use the following command:
    # im_ani.save('im.mp4', metadata={'artist':'Guido'})

    plt.show()


example_01()
# example_02()