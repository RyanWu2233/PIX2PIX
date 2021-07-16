# -*- coding: utf-8 -*-
"""
# Module name:      SIFT
# Author:           Ryan Wu
# Version:          V0.10- 2019/01/10
# Description:      Computer Vision Feature detector- SIFT
"""
import os,sys,inspect 
import numpy as np 
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

'================================================ SIFT ' 
class SIFT:
    """ 支援下列功能:
    KPDB,DESC= detect_and_compute(self,src_img) # 計算關鍵點, 描述子
    
    init_img = pre_processing(self,src_img);    # 圖片起始處理 
    
    img      = downsample_by2(ref):             # 向下採樣 (/2)    
    gauss_pyr= build_gauss_pyr(base):           # 建立高斯金字塔
    dog_pyr  = build_dog_pyr(gauss_pyr):        # 建立DOG金字塔 
        
    exts     = scale_space_extrema(dog_pyr):    # 求出關鍵點    
    feats    = calc_feature_oris(exts, gauss_pyr):    # 設定關鍵點方向    
               add_good_ori_features(feats,hist,mag_thr,ddata): # 設定關鍵點主方向
    hist     = ori_hist(img,ddata):             # 計算主方向分佈圖
        
    desc     = compute_descriptors(feats,gauss_pyr):  # 建立描述子     
    hist2    = descr_hist(img,ddata):                 # 產生2維描述子
               interp_hist_entry(hist2,rbin,cbin,obin,mag,d,n): # 登錄方向到描述子
    """
    def __init__(self):                         #---- 起始宣告
        # src_img: 原始圖片
        self.intvls         = 3;    # 影像金字塔每一層有幾個分頁
        self.sigma          = 1.6;  # 影像最開始的 Gaussian 平滑參數
        self.contr_thr      = 0.04; # 關鍵點提取的對比閥值
        self.curv_thr       = 10;   # 關鍵點提取的Hessian閥值
        self.descr_width    = 4;    # 關鍵點描述子的寬度
        self.descr_hist_bins= 8;    # 關鍵點描述子的方向解析度
        self.init_sigma     = 0.5;  # 輸入影像的 Gaussian 平滑參數
        self.img_border     = 5;    # 影像邊框的遮罩寬度 (像素)
        self.max_interp_steps=5;    # 關鍵點飄移計算的最大嘗試次數
        self.ori_hist_bins  = 36;   # 主方向計算時,分佈圖的bin數
        self.ori_sig_fctr   = 1.5;  # 主方向計算時,Gaussian 平滑參數
        self.ori_radius     = 3* self.ori_sig_fctr;
        self.ori_smooth_passes=2;   # 主方向計算,分佈圖的平滑次數
        self.ori_peak_ratio = 0.8;  # 多重主方向的判斷閥值
        self.descr_mag_thr  = 0.2;  # 影像亮度調整, 飽和閥值
        self.descr_scl_fctr = 3.0;  # 
        self.int_descr_fctr = 512.0;# 描述子的輸出 scaling (float--> int)  
        self.octvs_max      = 4;    # 最多幾層
        self.doubled        = False;# 是否要用雙倍大小影像

    '------------------------------------------------ Main function'
    def detect_and_compute(self,src_img):           #---- 計算關鍵點, 描述子
        print('---------------------')
        short_side = np.minimum(src_img.shape[0],src_img.shape[1]); # 圖片短邊
        self.octvs = np.floor(np.log2(short_side)-3).astype(int);   # 金字塔層數
        self.octvs = np.minimum(self.octvs,self.octvs_max);
        print('Build pyramid')
        init_img  = self.pre_processing(src_img);   # 圖片起始處理        
        gauss_pyr = self.build_gauss_pyr(init_img); # 建立高斯金字塔
        dog_pyr   = self.build_dog_pyr(gauss_pyr);  # 建立DOG金字塔 
        print('Detect keypoints')
        KPDB      = self.detect(gauss_pyr,dog_pyr); # 找尋關鍵點
        print('Compute descriptors')
        DESC      = self.compute(KPDB,gauss_pyr);   # 建立描述子
        print('Done')
        #---- 儲存計算結果 ----
        #FD_tools.plot_pyramid(gauss_pyr,figy=6);
        #FD_tools.plot_pyramid(np.abs(dog_pyr),figy=6, normalize= True);
        #FD_tools.plot_keypoints(src_img,KPDB,keytype='arrow')
        self.KPDB = KPDB;   
        self.DESC = DESC;
        self.dog_pyr  = dog_pyr;
        self.pyramid= gauss_pyr;
        return KPDB,DESC        
    
    '------------------------------------------------ Pre-processing'
    def pre_processing(self,src_img):
        # 執行以下動作: (1)轉灰階圖 (2)將資料改成[0~1] (3)影像放大2倍 (4)Gauss smooth
        gray    = CVP.rgb2gray(src_img);            # 灰階圖
        if np.max(gray)>2: gray= gray/256;          # 影像強度正規化到 [0 ~ 1]
        #doubled = CVP.resize(gray,ratio=2);         # 雙倍影像圖
        if self.doubled== True: doubled = CVP.resize(gray,ratio=2); # 雙倍影像圖
        else: doubled= gray;            
        sig_diff= np.sqrt(self.sigma**2- 4*self.init_sigma**2); # s=1.25
        init_img= CVP.Gauss_blur2D(doubled,sigma=sig_diff)
        return init_img

    '------------------------------------------------ Build pyramid'
    def build_gauss_pyr(self,base):                 #---- 建立高斯金字塔
        intvls    = self.intvls;                    # 影像金字塔每一層有幾個分頁 =3
        sigma     = self.sigma;                     # 影像最開始的 高斯平滑參數=1.6
        octvs     = self.octvs;                     # 高斯金字塔有幾層
        k         = 2**(1/intvls);                  # =2^(0.333)
        sig       = np.zeros(intvls+3);             # 每一頁的 sigma
        sig[0]    = sigma;
        sig[1]    = sigma*np.sqrt(k*k-1);
        for i in range(2,intvls+3): sig[i]= sig[i-1]*k;         
        gauss_pyr = [];                             # 輸出高斯金字塔
        for o in range(0,octvs):
          if o==0: 
            pyramid= np.zeros((base.shape[0],base.shape[1],intvls+3));
            pyramid[:,:,0]= base;
          else:
            pyramid= self.downsample_by2(gauss_pyr[o-1][:,:,intvls]);             
          for i in range(1,intvls+3):             
            pyramid[:,:,i]= CVP.Gauss_blur2D(pyramid[:,:,i-1],sigma=sig[i]);
          gauss_pyr.append(pyramid);        
        return gauss_pyr
    
    def downsample_by2(self,ref):                   #---- 向下採樣 (/2)
        rows      = ref.shape[0];                   # 原圖尺寸
        cols      = ref.shape[1];
        rowh      = np.ceil(rows/2).astype(int);    # 半圖尺寸
        colh      = np.ceil(cols/2).astype(int);
        halve     = np.zeros((rowh,colh,self.intvls+3));
        halve[:,:,0]= ref[0:rows:2,0:cols:2];
        return halve
    '------------------------------------------------ Differential of Gaussian'
    def build_dog_pyr(self,gauss_pyr):              #---- 建立DOG金字塔
        octvs     = self.octvs;                     # 高斯金字塔有幾層
        dog_pyr   = [];
        for o in range (0,octvs):
          rowp= gauss_pyr[o].shape[0];
          colp= gauss_pyr[o].shape[1];
          page= gauss_pyr[o].shape[2];
          dog = np.zeros((rowp, colp, page-1));
          for i in range(0,page-1): 
            dog[:,:,i]= gauss_pyr[o][:,:,i+1]- gauss_pyr[o][:,:,i];
          dog_pyr.append(dog);
        return dog_pyr

    '------------------------------------------------ 找尋關鍵點 '
    def detect(self,gauss_pyr,dog_pyr):             #---- 找尋關鍵點
        exts      = self.scale_space_extrema(dog_pyr);   # 求出關鍵點
        feats     = self.calc_feature_oris(exts, gauss_pyr); # 設定關鍵點方向
        return feats
    
    '------------------------------------------------ 找尋極點'
    def scale_space_extrema(self,dog_pyr):          #---- 求出極點
        sigma     = self.sigma;                     # 影像最開始的 高斯平滑參數=1.6
        octvs     = self.octvs;                     # 高斯金字塔有幾層
        intvls    = self.intvls;                    # 影像金字塔每一層有幾個分頁 =3
        border    = self.img_border;                # 影像邊界
        contr_thr = self.contr_thr;                 # 關鍵點提取的對比閥值 =0.04
        curv_thr  = self.curv_thr;                  # 關鍵點提取的Hessian閥值 =10
        curv2     = (1+curv_thr)**2;
        prelim_contr_thr = 0.5 *contr_thr / intvls; # DOG 影像強度閥值
        
        feat      = Keypoint();                     # 空白極點物件
        exts      = KP_DB();                        # 空白極點物件列表
        for o in range(0,octvs):                    # o:第幾層
          dog = dog_pyr[o];  
          for page in range(1,intvls+1):               # p: 第幾頁
            for y in range(border,dog.shape[0]-border):
              for x in range(border,dog.shape[1]-border):  
                is_ext = 0;                         # 1: 代表極點
                dogv= dog[y,x,page]; 
                if np.abs(dogv)> prelim_contr_thr:  #... 檢查對比與區域極值 ...
                  cube_27 = dog[y-1:y+2,x-1:x+2,page-1:page+2];
                  cube_max= np.max(cube_27);          # 3x3x3 的極大值
                  cube_min= np.min(cube_27);          # 3x3x3 的極小值                    
                  if dogv>0 and dogv== cube_max: is_ext=1;
                  if dogv<0 and dogv== cube_min: is_ext=1;               
                #--- 精準定位 ---
                if is_ext==1:                       # 進行主曲率檢測                    
                  r =y;  c=x;  p= page;             # 目前偵測點的位置
                  rr=y; cc=x; pp= page;             # 目前偵測點的位置
                  xr=0; xc=0; xp=0;                 # 偵測點的修正量
                  iter_times=0;                     # 修正次數
                  modify_flag=1;
                  while(modify_flag==1):
                    dy = (dog[r+1,c  ,p  ]- dog[r-1,c  ,p  ])*0.5;
                    dx = (dog[r  ,c+1,p  ]- dog[r  ,c-1,p  ])*0.5;
                    ds = (dog[r  ,c  ,p+1]- dog[r  ,c  ,p-1])*0.5;
                    dyy= (dog[r+1,c  ,p  ]+ dog[r-1,c  ,p  ]-dog[r,c,p]*2);
                    dxx= (dog[r  ,c+1,p  ]+ dog[r  ,c-1,p  ]-dog[r,c,p]*2);
                    dss= (dog[r  ,c  ,p+1]+ dog[r  ,c  ,p-1]-dog[r,c,p]*2);
                    dxy= (dog[r+1,c+1,p  ]+ dog[r-1,c-1,p  ]-dog[r+1,c-1,p  ]-dog[r-1,c+1,p  ])*0.25;
                    dys= (dog[r+1,c  ,p+1]+ dog[r-1,c  ,p-1]-dog[r+1,c  ,p-1]-dog[r-1,c  ,p+1])*0.25;
                    dxs= (dog[r  ,c+1,p+1]+ dog[r  ,c-1,p-1]-dog[r  ,c+1,p-1]-dog[r  ,c-1,p+1])*0.25;                 
                    
                    dD = np.matrix([[dx], [dy], [ds]]) 
                    H =  np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]]) 
                    
                    x_hat = np.linalg.lstsq(H, dD,rcond=None)[0]                     
                    xr = x_hat[0,0];  xc = x_hat[1,0];  xp=x_hat[2,0];    # 修正量
                    if np.abs(xr)<0.5 and np.abs(xc)<0.5 and np.abs(xp)<0.5: modify_flag=0;  # 修正完成 
                    #--- 計算新的特徵點位置 ---
                    rr += xr; r= np.round(rr).astype(int);
                    cc += xc; c= np.round(cc).astype(int);
                    pp += xp; p= np.round(pp).astype(int);
                    if r<border or r> dog.shape[0]-border: modify_flag= -1; # 超出範圍, 捨棄
                    if c<border or c> dog.shape[1]-border: modify_flag= -1; # 超出範圍, 捨棄
                    if p<1      or p> intvls:     modify_flag= -1; # 超出範圍, 捨棄
                    #--- ---
                    iter_times += 1; 
                    if iter_times>= self.max_interp_steps: modify_flag= -1; # 沒收斂, 捨棄
                  #---- --------------
                  if modify_flag==-1: is_ext=0;
                #--- 邊線檢查 ---
                if is_ext==1:
                  D_x_hat = dog[r,c,p] - 0.5* np.dot(dD.transpose(),x_hat);
                  tr = dxx+dyy; det= dxx*dyy-(dxy**2);    #--- 邊線排除(主曲率檢測) ---
                  if np.abs(D_x_hat)<(contr_thr / intvls): is_ext=0; # 排除不穩定點                  
                  if det<0:                                is_ext=0;
                  if (tr*tr*curv_thr)> (det*curv2):        is_ext=0; 
                #--- 資料登錄 ---
                if is_ext==1:  
                  if self.doubled== True: fd=2;
                  else: fd=1;
                  feat.y= rr* (2**o)/fd;  feat.r= r;
                  feat.x= cc* (2**o)/fd;  feat.c= c;
                  feat.octv    = o; 
                  feat.info[0] = p;                             # intvl
                  feat.info[1] = sigma*(2**(p/intvls+ o))/2;    # scl
                  feat.info[2] = sigma*(2**(p/intvls));         # scl_octv
                  exts.push(feat);      
        exts.fit(); 
        return exts
    '------------------------------------------------ 找尋主方向'
    def calc_feature_oris(self,exts, gauss_pyr):    #---- 設定關鍵點方向
        feats= KP_DB();                             # 空白關鍵點列表
        for n in range(0,exts.count):
          ddata= exts.get(n);                       # 讀取關鍵點            
          octv = (ddata.octv).astype(int);          # 該點在哪一層?
          hist = self.ori_hist(gauss_pyr[octv],ddata); # 計算主方向分佈圖
          omax = np.max(hist);                      # 計算主方向的bin
          mag_thr= omax*self.ori_peak_ratio;        # 接收比例 = 80% of peak
          self.add_good_ori_features(feats,hist,mag_thr,ddata);
        feats.fit();  
        return feats  
    
    def add_good_ori_features(self,feats,hist,mag_thr,ddata): #---- 設定關鍵點主方向
        n      = self.ori_hist_bins;                # 有幾個方位 (= 36)         
        pi2    = np.pi*2;   pi     = np.pi;   
        for i in range(0,n):
          hl= hist[np.mod(i-1, n)];
          hi= hist[i];
          hr= hist[np.mod(i+1, n)]; 
          if hi>hl and hi>hr and hi>mag_thr:        # 比左右大 = local peak
            bins= i+ 0.5*(hl-hr)/(hl+hr-2*hi);      # 相鄰項 二次修正
            bins= np.mod(bins,n);
            ddata.ori= (pi2*bins)/n- pi;
            #ddata.ori= (pi2*bins)/n;
            feats.push(ddata);  
        
    def ori_hist(self,img,ddata):                   #---- 計算主方向分佈圖
        intv  = ddata.info[0].astype(int);
        #scl   = ddata.info[1];
        scl_octv = ddata.info[2];
        
        rad   = np.round(self.ori_radius* scl_octv).astype(int);
        sigma =        self.ori_sig_fctr* scl_octv;
        exp_fa= 1/(2*sigma*sigma);                  # 高斯平滑參數
        pi    = np.pi;  pi2= np.pi*2;               # 圓周率
        n     = self.ori_hist_bins;                 # 分佈圖bin數目
        hist  = np.zeros(n);                        # 空白分佈圖
        rowp  = img.shape[0]-1;  colp= img.shape[1]-1; # 影像邊界
        r     = (ddata.r).astype(int);  
        c     = (ddata.c).astype(int);  
        #intv  = (intvl).astype(int);
        for i in range(-rad,rad+1):
          for j in range(-rad,rad+1):
            ri= r+i; cj= c+j;               
            if ri>0 and cj>0 and ri<rowp and cj<colp:
              dx  = img[ri  ,cj+1,intv] - img[ri  ,cj-1,intv];    # X 梯度
              dy  = img[ri-1,cj  ,intv] - img[ri+1,cj  ,intv];    # Y 梯度
              mag = np.sqrt(dx*dx+dy*dy);           # 梯度強度
              ori = np.arctan2(dy,dx);              # 梯度方向
              w   = np.exp(-(i*i+j*j)*exp_fa);
              bins= np.round(n*(ori +pi)/pi2);      # 累積權重
              bins= np.mod(bins,n).astype(int);
              hist[bins] += w* mag;
        #--- 平滑化處理 ---      
        hist_tmp= np.zeros(n);
        for smt in range(0,self.ori_smooth_passes):
          hist_tmp[:]     =0;  
          hist_tmp[:]    += hist[:]*0.5;
          hist_tmp[0:n-1]+= hist[1:n  ]*0.25; hist_tmp[n-1]+= hist[0  ]*0.25;
          hist_tmp[1:n  ]+= hist[0:n-1]*0.25; hist_tmp[0]  += hist[n-1]*0.25;
          hist[:]=hist_tmp;
        return hist

    '------------------------------------------------ 建立描述子'     
    def compute(self,feats,gauss_pyr):              #---- 建立描述子
        d      = self.descr_width;                  # 關鍵點描述子的寬度
        n      = self.descr_hist_bins;              # 關鍵點描述子的方向解析度
        desc   = np.zeros((feats.count,d*d*n));     # 建立空白描述子列表
        prog   = 0;
        for i in range(0,feats.count):
          ddata= feats.get(i);                      # 讀取關鍵點
          octv = (ddata.octv).astype(int);          # 該點在哪一層?
          hist2= self.descr_hist(gauss_pyr[octv],ddata); # 產生2維描述子
          vecf = np.reshape(hist2,(1,-1));          # [4,4,8] --> [1,128]
          norm = np.maximum(np.sqrt(np.sum(vecf*vecf)),0.01);
          vecf = np.minimum(vecf/norm,self.descr_mag_thr);
          norm = np.sqrt(np.sum(vecf*vecf));
          desc[i,:]= vecf/norm;                     # 登錄描述子
          if ((i+1)/feats.count)>=prog:                 # 顯示進度
            progx= np.floor(prog*100).astype(int);
            print('Progress= ',progx,' %'); 
            prog+=0.1;
        return desc
    
    def descr_hist(self,img,ddata):                 #---- 產生2維描述子
        #page   = (ddata.intvl).astype(int);         # 關鍵點位於哪個分頁
        #ori    = ddata.ori-np.pi;                   # 關鍵點 主方位
        #scl    = ddata.scl_octv;                    # 關鍵點 尺度
        page   = (ddata.info[0]).astype(int);       # 關鍵點位於哪個分頁
        ori    = ddata.ori;                         # 關鍵點 主方位
        scl    = ddata.info[2];                     # 關鍵點 尺度
        r      = (ddata.r).astype(int);             # 關鍵點 Row
        c      = (ddata.c).astype(int);             # 關鍵點 Col
        d      = self.descr_width;                  # 關鍵點描述子的寬度
        n      = self.descr_hist_bins;              # 關鍵點描述子的方向解析度
        hist2  = np.zeros((d,d,n));                 # 描述子輸出
        pi2    = np.pi*2;                           # 圓周率
        rows   = img.shape[0];  cols= img.shape[1]; # 圖片高度,寬度
        
        cos_t  = np.cos(ori);                       #
        sin_t  = np.sin(ori);                       #
        bins_per_rad= n/pi2;                        # 每個bin寬度= (8/6.283) 
        exp_denom   = d*d*0.5; 
        hist_width  = self.descr_scl_fctr* scl;     # 
        radius = hist_width* np.sqrt(2)* (d+1)*0.5+0.5; #
        radius = np.round(radius).astype(int);
         
        for i in range(-radius,radius+1):
          for j in range(-radius,radius+1):
            c_rot= (j*cos_t - i*sin_t) / hist_width;
            r_rot= (j*sin_t + i*cos_t) / hist_width;
            rbin = r_rot+ d/2 -0.5;
            cbin = c_rot+ d/2 -0.5;
            if rbin>-1 and rbin<d and cbin>-1 and cbin<d:
              ri= r+i; cj= c+j;  
              if ri>0 and cj>0 and ri<(rows-1) and cj<(cols-1):
                dx = img[ri,  cj+1,page] - img[ri,  cj-1,page];
                dy = img[ri-1,cj  ,page] - img[ri+1,cj  ,page];
                grad_mag= np.sqrt(dx*dx+ dy*dy);    # Local 梯度強度
                grad_ori= np.arctan2(dy,dx);        # Local 梯度方向
                grad_ori= np.mod(grad_ori-ori,pi2); # Local 梯度方向修正
                obin    = grad_ori* bins_per_rad;   # 屬於哪個bin
                w       = np.exp(-(c_rot*c_rot+ r_rot*r_rot)/exp_denom);
                self.interp_hist_entry(hist2,rbin,cbin,obin,grad_mag*w,d,n);        
        return hist2
    
    def interp_hist_entry(self,hist2,rbin,cbin,obin,mag,d,n): #---- 登錄方向到描述子
        r0 = np.floor(rbin).astype(int); dr= rbin- r0; r1= r0+1; 
        c0 = np.floor(cbin).astype(int); dc= cbin- c0; c1= c0+1; 
        o0 = np.floor(obin).astype(int); do= obin- o0;
        o0 = np.mod(o0,n);  o1=np.mod(o0+1,n);
        
        if r0>=0 and c0>=0:
          hist2[r0,c0,o0]+= mag*(1-dr)*(1-dc)*(1-do);
          hist2[r0,c0,o1]+= mag*(1-dr)*(1-dc)*(  do);
        if r0>=0 and c1< d:
          hist2[r0,c1,o0]+= mag*(1-dr)*(  dc)*(1-do);
          hist2[r0,c1,o1]+= mag*(1-dr)*(  dc)*(  do);
        if r1< d and c0>=0:
          hist2[r1,c0,o0]+= mag*(  dr)*(1-dc)*(1-do);
          hist2[r1,c0,o1]+= mag*(  dr)*(1-dc)*(  do);
        if r1< d and c1< d:
          hist2[r1,c1,o0]+= mag*(  dr)*(  dc)*(1-do);
          hist2[r1,c1,o1]+= mag*(  dr)*(  dc)*(  do);
          
    '------------------------------------------------ ' 
