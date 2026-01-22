# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 21:16:19 2024

@author: TM
"""

import zipfile
import os
os.chdir('C:/Users/tm/Downloads/utas/taselevation')

for i in os.listdir():
    os.makedirs(i[:-4])
    with zipfile.ZipFile(i, 'r') as zip_ref:
        zip_ref.extractall('C:/Users/tm/Downloads/utas/taselevation/'+i[:-4])