# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 00:21:37 2024

@author: TM
"""

import numpy as np
import pandas as pd
import os
os.chdir('C:/Users/tm/Downloads/utas/taselevation')


min_height=0
max_height=1600
interval=15

def asc2gpd(info, grid):

    
    #convert to int for lower left x,y, nodatavalue and cell size
    xllcorner=info['val'][info['name']=='xllcorner'].item()
    xllcorner=int(xllcorner)
    yllcorner=info['val'][info['name']=='yllcorner'].item()
    yllcorner=int(round(yllcorner))
    cellsize=info['val'][info['name']=='cellsize'].item()
    cellsize=int(cellsize)
    nodataval=info['val'][info['name']=='NODATA_value'].item()
    nodataval=int(nodataval)

    # Get coordinates using NumPy
    nrow = np.arange(yllcorner, yllcorner + cellsize * grid.shape[0], cellsize)
    ncol = np.arange(xllcorner, xllcorner + cellsize * grid.shape[1], cellsize)

    # Create coordinates array using NumPy
    xx, yy = np.meshgrid(ncol, nrow[::-1])
    coords = np.column_stack((xx.ravel(), yy.ravel()))

    # Remove nodata points
    height=grid.ravel()[grid.ravel()!=nodataval]
    coordinates=coords[grid.ravel()!=nodataval]    
    
    """
    #fix centroids for kmean
    centroids=np.arange(min_height,max_height,interval)
    distances = np.abs(height[:, np.newaxis] - centroids)
    
    # Find the index of the closest centroid for each data point
    labels = np.argmin(distances, axis=1)
    """    
    # Create GeoDataFrame directly from arrays
    output = pd.DataFrame({
        'easting': coordinates[:,0],
        'northing': coordinates[:,1],
        'altitude': height
    })
    
    return output

#iterations to read files
output=pd.DataFrame(columns=['northing','easting','altitude'])
for i in os.listdir():
    if 'LIST_DEM_25M' in i and 'zip' not in i:
        
        print(i)
        #read asc file into matrix
        grid=np.loadtxt(f'./{i}/{i.lower()}.asc', skiprows=6)

        #read the header of asc file to map the coordinates
        info=pd.read_csv(f'./{i}/{i.lower()}.asc',header=None, nrows=6)
        info['val']=info[0].apply(lambda x:x.split(' ')[-1]).astype(float)
        info['name']=info[0].apply(lambda x:x.split(' ')[0])

        print('convert to df')
        result=asc2gpd(info,grid)

        print('merge')
        #merge
        output=pd.concat([output,result])
        output.reset_index(inplace=True,drop=True)


print('export')
output.to_csv('./height.csv', index=False)

