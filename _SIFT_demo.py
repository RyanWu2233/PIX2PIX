# -*- coding: utf-8 -*-
"""
# Module name:      _SIFT_demo
# Author:           Ryan Wu
# Version:          V0.10- 2019/07/23
# Description:      Computer Vision signal processing:
                    demo file for SIFT 
"""
import os,sys,inspect 
import numpy as np
import matplotlib.pyplot as plt
#---------------------------------------------------
def insert_sys_path(new_path):                      #---- Insert new path to sys
  for paths in sys.path:
    path_abs = os.path.abspath(paths);
    if new_path in (path_abs, path_abs + os.sep): return 0
  sys.path.append(new_path); 
  return 1 
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) 
insert_sys_path(current_dir);
parent_dir = os.path.dirname(current_dir) 
insert_sys_path(parent_dir); 
#---------------------------------------------------
from P00_CVP.CVP import CVP 
from Features import Keypoint 
from Features import KP_DB  
from Features import FD_tools 
from SIFT import SIFT
#=================================================== Test bench
src1 = plt.imread('400_taipei1.jpg')/256;
src2 = plt.imread('400_taipei2.jpg')/256;
model1= SIFT();
model1.detect_and_compute(src1)
model2= SIFT();
model2.detect_and_compute(src2)
match= FD_tools.matching(model1.DESC, model2.DESC);

#--- show result ---
FD_tools.plot_pyramid(model1.pyramid,figy=8,ticks='off')
FD_tools.plot_keypoints(src1, model1.KPDB, keytype='arrow',figwidth=10, keysize=2)
FD_tools.plot_match(model1.KPDB, model2.KPDB, match, src1, src2, th=0.7)
FD_tools.plot_match(model1.KPDB, model2.KPDB, match, src1, src2, th=0.5)
FD_tools.listMatch(model1.KPDB, model2.KPDB, match, knn=True, lv=True, ori=True,pair_no=20)

#---------------------------------------------------

