# -*- coding: utf-8 -*-
"""
# Module name:      Features
# Author:           Ryan Wu
# Version:          V0.10- 2019/01/10
# Description:      Computer Vision Feature detector
"""
#---------------------------------------------
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
'================================================ Keypoint '
class Keypoint:                                 #---- 關鍵點資料
    def __init__(self,Dtype='SIFT'):
        self.y       = 0;           # 對應到原始影像的 Y 座標 [pixel]
        self.x       = 0;           # 對應到原始影像的 X 座標 [pixel]
        self.r       = 0;           # 對應到該分層的 Y 座標 [pixel]
        self.c       = 0;           # 對應到該分層的 X 座標 [pixel]
        self.octv    = 0;           # 分層 [0 ~ octvs]
        self.ori     = 0;           # 方向 [0 ~ 2*pi]
        self.Dtype   = Dtype;       # 特徵點方法
        self.info    = np.zeros(10);# 額外資料
                                    # SIFT :[intvl,scl,scl_octv]
                                    # ORB  :[Fscore,Hscore]                                    
        
'================================================ Keypoint database ' 
class KP_DB:                                    #---- 關鍵點資料庫
    """ 
    關鍵點資料庫:
    keypoint 的格式: [y,x, r,c, octv, ori, data]
    支援下列功能:
        
            linkDB(ORB_KP_DB) : 加入另一個資料庫    
            push(ORB_KP)      : 將ORB_KP的資料填入列表
            replace(no,ORB_KP): 將列表內第no筆ORB_KP 更新
            fit()             : 將多餘資料去除
    feat  = get(no)           : 讀取列表內第no筆ORB_KP
    result= search(ymin,ymax,xmin,xmax): 將區域內的關鍵點找出
                        輸出列表的格式為: [編號,Y,X,大小,方向]
    """    
    def __init__(self,Dtype='SIFT'):
        self.count   = 0;                           # 登錄到第幾筆
        self.seg     = 500;                         # 1個block 有幾筆資料
        self.Dtype   = Dtype;                       # 額外資訊的長度;
        self.keyz    = np.zeros((self.seg,16));     # 座標, 尺寸, 角度
    
    def copy(self,KPZ):
        self.count   = KPZ.count;
        self.seg     = KPZ.seg;
        self.Dtype   = KPZ.Dtype;
        self.keyz    = KPZ.keyz;
        
    def linkDB(self,KPZ):                           #-- 加入另一個資料庫
        self.count  += KPZ.count;
        self.keyz    = np.concatenate((self.keyz,KPZ.keyz),axis=0);
        
    def replace(self,no,kp):                        #-- 修改資料
        self.keyz[no,0:6] = [kp.y, kp.x, kp.r, kp.c, kp.octv, kp.ori];
        self.keyz[no,6:16]=  kp.info;
          
    def push(self,kp):                              #-- 填入新資料
        self.replace(self.count, kp);
        self.count +=1;
        if self.count>= self.keyz.shape[0]:         # 資料超過表格大小, 追加配置記憶體
          self.keyz = np.concatenate((self.keyz ,np.zeros((self.seg,16))),axis=0);
 
    def get(self,no):                               #-- 用編號查詢資料
        res= Keypoint();
        res.y    = self.keyz[no,0];
        res.x    = self.keyz[no,1];
        res.r    = self.keyz[no,2];
        res.c    = self.keyz[no,3];
        res.octv = self.keyz[no,4];
        res.ori  = self.keyz[no,5];
        res.info = self.keyz[no,6:16];
        res.Dtype= self.Dtype;
        return res
    
    def fit(self):                                   #-- 去除多餘資料
        self.keyz = self.keyz[ 0:self.count,:];

    def area_search(self,area):                     #-- 區域搜尋
        # 傳回 [No, Y,X,scale,orientation]
        res = np.zeros((self.count,5)); m=0;
        ymin=area[0]; ymax=area[1]; xmin=area[2]; xmax=area[3]; 
        print('  no   y   x octv  ori')
        print('-------------------------------')
        for k in range(0,self.count):
          y = self.keyz[k,0];
          x = self.keyz[k,1];
          if x>xmin and x<xmax and y>ymin and y<ymax:
            res[m,:]=[k,y,x,self.keyz[k,4],self.keyz[k,5]];
            txt  = ' %3d %3d %3d ' % (res[m,0],res[m,1],res[m,2]);
            txt += ' %2d %6.2f ' % (res[m,3],res[m,4]*57.2958);
            print(txt)   
        if m>0: res= res[0:m,:];                    # Remove reduntant
        return res
     
'================================================ Feature detector tools ' 
class FD_tools:                                 #---- 特徵點偵測工具庫
    def plot_pyramid(pyr,start=0,end=0,**kwargs):
        """
        @ src  : 影像金字塔
        @ start: 從第幾層開始
        @ end  : 到弟幾層
        @ figx,figy: 影像大小
        """
        layers= np.size(pyr);                   # 金字塔有幾層
        if end== 0: end= layers;                # 最後第幾層        
        if pyr[0].ndim==2: pages= 1;            # 每一層有幾個分頁
        if pyr[0].ndim==3: pages= pyr[0].shape[2]; 
        ymax  = 0; xmax=0;
        ys    = np.zeros(end-start);            # 每一層的起始位置Y
        #---- ----
        for octv in range(start,end):           #-- 計算位置
          ys[octv-start]= ymax;
          ymax+= pyr[octv].shape[0];
          xmax = np.maximum(xmax,pyr[octv].shape[1]*pages);
        dest   = np.zeros((ymax,xmax));        
        #---- ----
        for octv in range(start,end):           #-- 填入資料
          base = pyr[octv];
          rows = base.shape[0]; 
          cols = base.shape[1]; 
          for page in range(0,pages):
            if pages>1 : img = base[:,:,page];
            if pages==1: img = base[:,:]
            y1  = ymax-(ys[octv-start]).astype(int)-rows; 
            x0  = page*cols; 
            dest[y1:y1+rows,x0:x0+cols]= img;        
        if kwargs.get('normalize'):             #-- 調整動態範微
          dmax= np.max(dest);  dmin= np.min(dest);
          dest= (dest-dmin)/(dmax-dmin);
          dmax= np.max(dest);  dmin= np.min(dest);          
        CVP.imshow(dest,**kwargs); plt.show();  # 顯示金字塔圖
     
    '------------------------------------------------ 標示關鍵點 '
    def plot_keypoints(img,KPDB,**kwargs):
        """
        在圖上畫出關鍵點
        @ img       : 原始圖片
        @ KPDB      : 關鍵點資料庫
        @-----------
        @ brightness: 原始圖片強度 [0~1]; default=0.7;
        @ keycolor  : 關鍵點顏色: 'c','r','b','w','g'; default= 'y'
        @ keytype   : 關鍵點形狀: 'circle'= 圓圈, 'arrow'= 箭頭        
        @ keysize   : 關鍵點大小: 1= Normal; 
        @ strength  : 關鍵點指標: 'level'=第幾層, 'Fscore'= FAST score, 'Hscore'=Harris score
        @ figwidth  : 圖片寬度: (in inches), default= 12;
        """
        brightness= 0.7;
        keycolor  = 'y';
        keytype   = 'circle';
        keysize   = 1;
        strength  = 'level';
        figwidth  = 8;
        #---- 讀取輸入參數 ----        
        if kwargs.get('brightness'):brightness=kwargs.get('brightness')
        if kwargs.get('keycolor'):  keycolor=  kwargs.get('keycolor')
        if kwargs.get('keytype'):   keytype=   kwargs.get('keytype')
        if kwargs.get('keysize'):   keysize=   kwargs.get('keysize') 
        if kwargs.get('strength'):  strength=  kwargs.get('strength')
        if kwargs.get('figwidth'):  figwidth=  kwargs.get('figwidth') 
        #---- 計算資料 ----  
        x     = KPDB.keyz[:,1];             # 關鍵點 X 座標
        y     = KPDB.keyz[:,0];             # 關鍵點 Y 座標
        xinch = figwidth;
        yinch = xinch*img.shape[0]/img.shape[1];        
        #-- 計算圓圈大小, arrow 長度---
        stre= (KPDB.keyz[:,4])              # Default 使用 level
        # if strength == 'level':  stre= KPDB.keyz[:,4]
        if strength == 'Fscore': stre= KPDB.keyz[:,6]
        if strength == 'Hscore': stre= KPDB.keyz[:,7]
        stre_max= np.max(stre);
        stre_min= np.min(stre);
        if stre_max== stre_min: stre_max= stre_max+1;
        stre    = 0.2+(stre-stre_min)/(stre_max - stre_min)*0.8; # Normalize to [0~1]
        streng  = pow(1.4,stre*10)*25*keysize;
        #---- 計算arrow  ----
        ori     = KPDB.keyz[:,5];  
        dy      = np.sin(ori)*streng/120; 
        dx      = np.cos(ori)*streng/120;
        if KPDB.Dtype=='SIFT': dx=-dx;         
        #---- 畫圖 ----
        plt.figure(figsize=(xinch,yinch));  # 調整圖片大小
        plt.imshow(img*brightness);         # 顯示原圖
        if keytype=='circle': plt.scatter(x,y,s= streng, facecolors='none', edgecolors= keycolor)
        if keytype=='arrow':  
          for k in range(0,KPDB.count):    
            plt.arrow(x[k],y[k],dx[k],dy[k],width=0.5,color=keycolor)              
        plt.show();  
    '------------------------------------------------ 顯示匹配圖 '
    def plot_match(KPDB1,KPDB2,match,img1,img2,**kwargs):    
        """
        @ img1, img2  : 要匹配的兩張圖
        @ KPDB1,KPDB2 : 關鍵點資料庫 
        @ match       : 匹配結果
        @-------------
        @ brightness  : 原始圖片強度 [0~1]; default=0.7;        
        @ height      : 圖片高度: (in inches), default= 6;
        @ th          : 顯示配對的距離比 (knn1/knn2)
        @ pair_no     : 顯示配對數
        @ full        : 顯示全部配對 (full= True)
        @ markersize  : 標記點大小 (default= 5)  
        @ linewidth   : 線條寬度 
        @ line_off    : 不顯示線條
        """
        #--- 預設參數 ---
        brightness= 0.7;                            # 背景圖的亮度 [1為原亮度]
        height    = 6;                              # 圖片高度
        th        = 1;                              # Ratio threshold
        color_seq = ['b','g' ,'r','c','m','y','k','w'];
        mark_seq  = ['o','s' ,'^','v','+','x','*'];
        line_seq  = ['-','--',':',':' ,':' ,'' ,'' ];
        markersize= 6;                              # 標記大小 
        pair_no   = np.minimum(10,match.shape[0]);  # 預設顯示前10 配對
        line_off  = 0;
        #--- 讀取輸入參數---
        if kwargs.get('brightness'): brightness= kwargs.get('brightness')        
        if kwargs.get('height'):     height    = kwargs.get('height')
        if kwargs.get('markersize'): markersize= kwargs.get('markersize');
        
        if kwargs.get('full'):       pair_no   = match.shape[0];
        if kwargs.get('th'):         pair_no   = match.shape[0]; th=kwargs.get('th'); 
        if kwargs.get('pair_no'):    pair_no   = kwargs.get('pair_no');
        if kwargs.get('line_off'):   line_off  = 1;
        #--- ---
        row1 = img1.shape[0]; col1 = img1.shape[1];
        row2 = img2.shape[0]; col2 = img2.shape[1];
        #--- 顯示背景圖片 ---
        dest = np.zeros((np.maximum(row1,row2),col1+col2,3));     # 輸出圖片        
        dest[0:row1,0:col1,:]        = img1;
        dest[0:row2,col1:col1+col2,:]= img2;
        if np.max(dest)>1: dest=dest/256;
        width = height*(dest.shape[1]/dest.shape[0]);
        plt.figure(figsize=(width,height));         # 調整圖片大小
        plt.imshow(dest*brightness);                # 顯示背景圖
        #--- 顯示配對 ---
        list_no  = 0;                               # 已經顯示幾點
        for k in range(0,pair_no):
          m   = match[k,:];
          if list_no<pair_no and m[2]<th:
            bin1= m[0].astype(int);                 # 圖1對應關鍵點
            bin2= m[1].astype(int);                 # 圖2對應關鍵點
            y1  = KPDB1.keyz[bin1,0];               # 圖1對應關鍵點座標
            x1  = KPDB1.keyz[bin1,1];
            y2  = KPDB2.keyz[bin2,0];               # 圖2對應關鍵點座標
            x2  = KPDB2.keyz[bin2,1]+col1;
            clr = color_seq[np.mod(list_no,8)];     # 標記/線條顏色
            r   = np.minimum(np.floor(list_no/8),6).astype(int);# 第幾層
            mark= mark_seq[r];                      # 標記點
            ls  = line_seq[r];                      # 線條
            #ls='-'
            if line_off==1: ls='';
            pattern= clr+ ls+ mark;
            plt.plot([x1,x2],[y1,y2],pattern,markersize=markersize); 
            list_no +=1;             
        #--- ---
        plt.show();
 
    '------------------------------------------------ 列出匹配資訊 '
    def listMatch(KPDB1,KPDB2,match,**kwargs):
        """
        @ KPDB1,KPDB2 : 關鍵點資料庫 
        @ match       : 匹配結果
        @-------------
        @ full        : 顯示全部配對
        @ pair_no     : 顯示配對數
        @ th          : 顯示配對的距離比 (knn1/knn2)
        @ knn=True    : 列出knn資訊:  [ratio, knn1, knn2]
        @ xy =True    : 列出XY座標資訊: [level1, level2]
        @ lv =True    : 列出level資訊: [level1, level2]
        @ ori=True    : 列出角度資訊: [ori1, ori2] 
        """
        th        = 1;                              # Ratio threshold
        pair_no   = 10;                             # 預設顯示前10 配對
        #--- 讀取輸入參數---
        if kwargs.get('full'):    pair_no = match.shape[0];
        if kwargs.get('th'):      pair_no = match.shape[0]; th=kwargs.get('th'); 
        if kwargs.get('pair_no'): pair_no = kwargs.get('pair_no');
        #--- 列出資訊標題 ---
        title= '  no bin1 bin2 ';  
        if kwargs.get('knn'): title += 'ratio  knn1  knn2 '; 
        if kwargs.get('lv'):  title += 'lv1 lv2 ';
        if kwargs.get('ori'): title += 'ori1 ori2 ';
        print(title); 
        print('-------------------------------------------------')
        #--- 列出資訊 ---
        list_no  = 0;                               # 已經顯示幾排         
        for k in range(0,match.shape[0]):
          m = match[k,:];            
          if list_no<pair_no and m[2]<th:  
            db1 = KPDB1.keyz[m[0].astype(int),:];
            db2 = KPDB2.keyz[m[1].astype(int),:];
            txt = '%4d %4d %4d '  % (list_no,m[0],m[1])              
            if kwargs.get('knn'): txt += ' %.2f  %4.2f  %4.2f '  % (m[2],m[3],m[4]); 
            if kwargs.get('lv') : txt += '%3d %3d ' % (db1[4],db2[4]);            
            if kwargs.get('ori'): txt += '%4d %4d ' % (db1[5]*57.3,db2[5]*57.3);
            print(txt);
            list_no +=1;
    '------------------------------------------------  '
    '------------------------------------------------  圖片匹配 '
    def matching(DESC1,DESC2,th=0.7, Dtype='SIFT'):
        """
        @ img : 原始圖片
        @ KPDB: 關鍵點資料庫
        @ DESC: 描述子資料庫
        輸出 match 矩陣: [bin1,bin2,ratio, dist1,dist2,TF]
        TF: 0=無資料; 1=True; -1=False
        """
        print('begin matching...')
        size1  = DESC1.shape[0];                    # 圖1有幾個特徵點
        nbr_bin= np.zeros((size1,2),dtype=int);     # 編號: 0:最近鄰; 1:第二近鄰
        nbr_dis= np.zeros((size1,2));               # 距離: 0:最近鄰; 1:第二近鄰
        
        dist   = FD_tools.compute_similarity(DESC1,DESC2,Dtype=Dtype); #計算相似性
        # plt.imshow(dist) 
        distmap= np.zeros_like(dist); distmap[:,:]= dist;# 複製一份資料
        distmax= np.max(dist);                      # 最長距離
        seq    = np.arange(0,size1);
        #---- 最近鄰 ----
        nbr_bin[:,0] = np.argmin(distmap,axis=1);   # 找出最近鄰編號
        nbr_dis[:,0] = distmap[seq,nbr_bin[:,0]];   # 登錄最近鄰距離
        distmap[seq,nbr_bin[:,0]]= distmax;         #     
        #---- 第二鄰 ----
        nbr_bin[:,1] = np.argmin(distmap,axis=1);   # 找出第二近鄰編號
        nbr_dis[:,1] = distmap[seq,nbr_bin[:,1]];   # 登錄第二近鄰距離
        #---- 篩選合適點---- 
        nbr_rat= nbr_dis[:,0]/ nbr_dis[:,1];        # = 最近鄰距離/ 第二近鄰距離
        cand_no= np.sum(nbr_rat<th);                # 符合條件的配對總數
        match  = np.zeros((cand_no,6));             # 匹配結果
        for k in range(0,cand_no):                  # 登錄配對結果
          bin1 = np.argmin(nbr_rat);
          bin2 = nbr_bin[bin1,0];
          match[k,0]= bin1;                         # 登錄 bin1
          match[k,1]= bin2;                         # 登錄 bin2
          match[k,2]= nbr_rat[bin1];                # 登錄距離
          match[k,3]= dist[bin1,bin2];
          match[k,4]= dist[bin1,nbr_bin[bin1,1]];
          nbr_rat[bin1]=1;                          # 已經配對完, 移除出比較序列
        return match  
    
    def compute_similarity(DESC1,DESC2,Dtype='SIFT'):   #---- 計算相似性
        size1= DESC1.shape[0];                  # 圖 1有幾個特徵點
        size2= DESC2.shape[0];                  # 圖 2有幾個特徵點
        dist = np.zeros((size1,size2));
        for k1 in range(0,size1):
          for k2 in range(0,size2):
            vec = DESC1[k1,:]-DESC2[k2,:];
            dist[k1,k2] = np.sqrt(np.sum(vec*vec));             
        return dist 
    
    '------------------------------------------------  '
 



    
    
    
    