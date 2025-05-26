#%matplotlib inline
import matplotlib.pyplot as plt

def plot_traces(traces):
    """
    Simple plot a bunch of power traces. 

    traces: power traces as ndarray of shape (ntraces, nsamples)
    """
    # Create the figure and plot the loaded traces
    f = plt.figure()
    ax0 = f.add_subplot(1,1,1)
    ax0.plot(traces.T)
    ax0.set_xlabel("Time index")
    ax0.set_ylabel("Power")
    f.tight_layout()
    plt.show()
