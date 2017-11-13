import seaborn as sns
sns.set_style("white") #whitegrid, darkgrid
sns.set_context("talk") #paper, talk, poster,

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
mpl.rc("figure", facecolor="white")

from matplotlib.pylab import rcParams

def set_fig_style(w=5.0, h=4.0, lsize=15, fsize=15, grid=True):
    """ set figure properties """
    rcParams['figure.figsize'] = w, h
    rcParams['xtick.labelsize'] = lsize
    rcParams['ytick.labelsize'] = lsize
    rcParams['axes.labelsize'] = lsize
    rcParams['legend.fontsize'] = fsize
    rcParams['axes.grid'] = grid
    rcParams['grid.linestyle'] = ':'
    rcParams['grid.linewidth'] = 1.0
    
def get_gridspec(w, h, w_margin, h_margin):
    gs = gridspec.GridSpec(w, h)
    gs.update(wspace=w_margin, hspace=h_margin)
    return gs
        
def get_col_palette(name='deep', n=10):
    """ return sns color palette 
    
    # Arguments:
        name: deep, muted, pastel, bright, dark, colorblind
        n: number of colors to be returned
    """
    return sns.color_palette(name, n)

def get_markers():
    return ['o', 's', 'D', '*', 'v', '^', 'p', 'h', 'x', '+']