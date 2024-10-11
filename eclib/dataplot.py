import matplotlib.pyplot as plt

def base_timeseries(figsize=(15,5)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()

    return fig, ax



def plot_timeseries(xs=None, ys=None, labels=None, clrs=None, title=None, ylim=None, xlabel=None, ylabel=None,
                    log=None, ncol=1, loc='upper right', textsize=15, filename=None, show=True):
    
    fig, ax = base_timeseries()

    if ys is None:
        ys = xs
        xs = None
    if not isinstance(ys, list): ys = [ys] 
    
    if xs is None:
        xs = [None]*len(ys)
    if not isinstance(xs, list): xs = [xs]    
        
    if labels is None:
        labels = [None]*len(ys)
        legend = False
    else: 
        legend = True

    if clrs is None:
        clrs = [None]*len(ys)

    if not isinstance(labels, list): labels = [labels]
        
    for x, y, label, clr in zip(xs, ys, labels, clrs):
        print(label)
        if x is None:
            ax.plot(y, label=label, linewidth=2, color=clr)
        else:
            ax.plot(x, y, label=label, linewidth=2, color=clr)

    if ylim: 
        ax.set_ylim(ylim)
        
    if legend: 
        ax.legend(fontsize=textsize-2, loc=loc, ncol=ncol)
        
    if log: 
        ax.set_yscale('log')
    
    ax.grid()
    ax.set_title(title, fontsize=textsize)
    ax.set_ylabel(ylabel, fontsize=textsize)
    ax.set_xlabel(xlabel, fontsize=textsize)
    ax.tick_params(axis='both', labelsize=textsize)

    if filename:
        plt.savefig(filename, format='png', dpi=50, bbox_inches='tight')
    
    if show:  
        plt.show()
        
    plt.close()