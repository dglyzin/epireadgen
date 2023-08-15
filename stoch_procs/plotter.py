import matplotlib.pyplot as plt


def plot(name, xs, titles, dim=1):
    count = len(xs)
    if count == 1:
        if dim == 1:
            r = plt.plot(xs[0])
        elif dim == 2:
            r = plt.plot(xs[0][:, 0], xs[0][:, 1])
        else:
            plotter_exception(dim)
            
        r[0].figure.savefig(name+".jpg")
    else:
        fig, axs = plt.subplots(count, 1)
        for idx, x in enumerate(xs):

            if dim == 1:
                axs.flat[idx].plot(x)
            elif dim == 2:
                axs.flat[idx].plot(x[:, 0], x[:, 1])
            else:
                plotter_exception(dim)
                
            axs.flat[idx].set_title(titles[idx])        

        fig.savefig(name+".jpg")

        
def plotter_exception(dim):
    raise(Exception("plotter: dim '%d' is not supported " % dim))
