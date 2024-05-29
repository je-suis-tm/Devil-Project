# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 21:11:18 2024

@author: TM
"""

import os
import geopandas as gpd
import pandas as pd
os.chdir('C:/Users/tm/Downloads/utas/tasveg')

grande=pd.DataFrame(columns=['VEG_GROUP', 'geometry', 'VEGCODE', 'FOREST_STR', 'CANOPY_TRE',
       'SOURCE_TYP', 'SOURCE_DAT', 'SOURCE_IRS', 'FIELD_CHEC', 'PROJECT',
       'SHAPE_LENG', 'VEGCODE_D', 'SHAPE_AREA', 'SHAPE_LEN'])

for i in os.listdir():
    if 'LIST_TASVEG_40' in i and 'zip' not in i:
        
        print(i)
        zip1=gpd.read_file(f'./{i}/{i.lower()}.shp')
        
        print('append')
        grande=pd.concat([grande,zip1])
    

print('aggregated')

print('convert to gpd')
grande=gpd.GeoDataFrame(grande, geometry='geometry')
        
print('dissolve')
grande=grande.dissolve(by='VEG_GROUP')
        
grande=grande.reset_index()

print('change coordinate system')
grande=grande.to_crs('epsg:4326')
print('export')
grande.to_file('./aggr.shp', driver='ESRI Shapefile')