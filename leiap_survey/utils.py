import numpy as np
import rasterio as rio
import matplotlib as mpl
import matplotlib.pyplot as plt

''' Function utilities'''
name = 'utilities'

def read_raster(fn):
    '''
    Reads raster into a 2D numpy array

    Parameters
    ----------
    fn: string
        path to raster image (assume geotiff)

    Return
    -------
    ras: 2D numpy array
        raster
    profile: dictionary
        raster geospatial information

    '''
    try:
        with rio.open(fn) as src:
            profile = src.profile
            ras = src.read(1)
            # change nodata value
            ras[ras == profile['nodata']] = -9999

            # make raster c-contiguous
            ras = np.ascontiguousarray(ras, dtype=np.float64)
            profile['dtype']= 'float64'
            profile['bounds'] = src.bounds
            return ras, profile

    except EnvironmentError:
        print('Oops! Could not find file')


def plot_raster(img, colors='viridis', title= None, figsize=(10, 10), ax=None,
                nodata_color={'color':'k', 'alpha': 0.0}):
    '''
    Simple function to plot raster and returns axes

    Parameters
    ----------
    img: rasterio raster
        raster image we want to plot
    colors: string
        name of any matplotlib colormap
    title: string
        Title use for raster. Default: None
    figsize: tuple
        Size of plot. Default: (10,10)
    ax: axes
        Matplotlib axes. Default: None
    no_data: dictionary
        Use 'color' and 'alpha' to specify the color and transparency
        associated with nodata

    Return
    ------
        Matplotlib axes
    '''

    from copy import copy

    # new figure?
    if not ax:
        # generate figure and axes
        fig, ax = plt.subplots(figsize=figsize)

    # unpack background image information
    ras = img.read(1)
    ras[ras == img.profile['nodata']] = -9999
    ras = np.ascontiguousarray(ras, dtype=np.float64)
    img.profile['dtype'] = 'float64'
    bounds = img.bounds
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    ax.ticklabel_format(axis='both', style='plain')
    ax.tick_params(color='slategrey', labelcolor='slategrey')
    ax.set_frame_on(False)

    # color
    cmap = mpl.cm.get_cmap(colors)

    # nodata?
    cmap_nodata = copy(cmap)
    if np.any(ras == -9999):
        ras = np.ma.array(ras, mask= ras == -9999)
        cmap_nodata.set_bad(nodata_color['color'], nodata_color['alpha'])

    # plot image
    _ = ax.imshow(ras, extent=extent, origin='upper', cmap=cmap_nodata)
    ax.set_title(title, fontsize=20)

    return ax
