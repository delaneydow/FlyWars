# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 12:13:01 2025

@author: laney
"""

# PULL IMPORTS
from optotune_sdk import Mirror
import time

# CONNECT MIRROR / RECOGNIZE
mirror = Mirror()
mirror.connect()  # auto-detects via USB


# ITERATE THROUGH GRID CALIBRATION & RECORD TIME TO CHANGE POSITIONS
# Move mirror to a few test positions
mirror.set_angle(x=0.0, y=0.0)
print("x=0, y=0 successful")
mirror.set_angle(x=1.0, y=0.0)
print("x=1, y=0 successful")
mirror.set_angle(x=0.0, y=1.0)
print("x=0, y=1 successful")
mirror.set_angle(x=-1.0, y=-1.0)
print("x=1, y=1 successful")

# RANDOM CALIBRATION

mirror.disconnect()