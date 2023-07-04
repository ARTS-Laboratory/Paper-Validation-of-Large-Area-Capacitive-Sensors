# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:02:53 2020

@author: AVEREEN
"""

import IPython as IP
IP.get_ipython().magic('reset -sf')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import matplotlib.colors as clr

from matplotlib import rcParams
plt.rcParams.update({'image.cmap': 'viridis'})
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.serif':['Times New Roman', 'Times', 'DejaVu Serif',
 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
 'Century Schoolbook L',  'Utopia', 'ITC Bookman', 'Bookman', 
 'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
plt.rcParams.update({'font.family':'serif'})
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.fontset': 'custom'})


rc('font',**{'family':'serif','serif':['Times New Roman']})
#rc('text', usetex=True)


plt.close('all')

xl="impact energy " + r"$\left( \mathrm{J} \right)$"
yl="capacitance change " + r"$\left( \frac{\mathrm{pF}}{\mathrm{pF}} \right)$"
yl1="capacitance " + r"$\left( \mathrm{pF} \right)$"



def moving_average(x, w):
    import numpy as np
    return np.convolve(x, np.ones(w), 'valid') / w

def strip_plot(du,tt,fi,la): # numpy
    import numpy as np
    loc0=np.where(tt>=fi)
    loc0=loc0[0]
    loc0=int(loc0[0])
    loc1=np.where(tt<=la)
    loc1=loc1[0]
    loc1=int(loc1[-1])
    n=loc1-loc0
    utt=np.arange(0,(la-fi),(la-fi)/(n))
    loc=loc0
    udu=[]
    while loc <= loc0+n-1:
        udu.append(du[loc])
        loc+=1    
    if len(utt) > len(udu):
        udu.append(du[loc])

    return udu , utt

def plot_this_too(TT,DU,xl,yl1,yl2,limx,i,N):
    ttCAP=np.asarray(TT[2*i])
    duCAP=np.asarray(DU[2*i])
    sx0=float(limx[0,2*i])
    sx1=float(limx[1,2*i])
    duC ,ttC = strip_plot(duCAP,ttCAP,sx0,sx1)
    ttLOAD=np.asarray(TT[2*i+1])
    duLOAD=np.asarray(DU[2*i+1])
    lx0=float(limx[0,2*i+1])
    lx1=float(limx[1,2*i+1])
    duL ,ttL = strip_plot(duLOAD,ttLOAD,lx0,lx1)
    fig = plt.figure(figsize=(6.5,4))
    ax1 = fig.add_subplot()
    title = '%s cm Impact trial'%(N[2*i+1])
    plt.title(title)
    
    color = 'tab:red'
    ax1.set_xlabel(xl)
    ax1.set_ylabel(yl1, color=color)
    ax1.plot(ttL, duL ,'-.', color=color, label = yl1)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  
    
    color = 'tab:blue'
    ax2.set_ylabel(yl2, color=color) 
    ax2.plot(ttC, duC ,'--', color=color, label = yl2)
    ax2.tick_params(axis='y', labelcolor=color)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=4)
    
    plt.savefig(title+'cm.png', dpi=500)
    plt.tight_layout()
    return fig

def plot_this(x,y,xl,yl):
    fig=plt.figure(figsize=(6.5,4))
    for i in range(0,len(x)):
       plt.plot(x[i],y[i],'b^')
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.tight_layout()
    plt.savefig('cmp.pdf')
    return fig

#%% load data

C=[]
L=[]
N=[]
MASSIMP=7.4752023
SEC=['SEC29_20CM_N2/cap.lvm',
     'SEC27_20CM_N4/cap.lvm',
     'SEC25_18CM_N6/cap.lvm',
     'SEC24_15CM_N7/cap.lvm',
     'SEC23_15CM_N8/cap.lvm',
     'SEC22_12.5CM_N10/cap.lvm',
     'SEC21_12.5CM_N11/cap.lvm',
     'SEC19_10CM_N12/cap.lvm',
     'SEC20_10CM_N13/cap.lvm',
     'SEC18_25CM_N14/cap.lvm',
     'SEC16_25CM_N15/cap.lvm',
     'SEC17_22.5CM_N16/cap.lvm',
     'SEC15_22.5CM_N17/cap.lvm',
     'SEC14_27.5CM_N18/cap.lvm',
     'SEC13_27.5CM_N19/cap.lvm',
     'SEC12_25CM_N20/cap.lvm',
     'SEC11_25CM_N21/cap.lvm',
     'SEC10_30CM_N22/cap.lvm',
     'SEC9_30CM_N23/cap.lvm',
     'SEC8_27.5CM_N24/cap.lvm',
     'SEC7_27.5CM_N25/cap.lvm',
     'SEC6_25CM_N26/cap.lvm',
     'SEC5_25CM_N27/cap.lvm',
     'SEC4_25CM_N28/cap.lvm',
     'SEC3_22.5CM_N29/cap.lvm',
     'SEC2_25CM_N30/cap.lvm',
     'SEC1_25CM_N31/cap.lvm']


LOAD=['SEC29_20CM_N2/force.lvm',
      'SEC27_20CM_N4/force.lvm',
      'SEC26_18CM_N6/force.lvm',
      'SEC24_15CM_N7/force.lvm',
      'SEC23_15CM_N8/force.lvm',
      'SEC22_12.5CM_N10/force.lvm',
      'SEC21_12.5CM_N11/force.lvm',
      'SEC19_10CM_N12/force.lvm',
      'SEC20_10CM_N13/force.lvm',
      'SEC18_25CM_N14/force.lvm',
      'SEC16_25CM_N15/force.lvm',
      'SEC17_22.5CM_N16/force.lvm',
      'SEC15_22.5CM_N17/force.lvm',
      'SEC14_27.5CM_N18/force.lvm',
      'SEC13_27.5CM_N19/force.lvm',
      'SEC12_25CM_N20/force.lvm',
      'SEC11_25CM_N21/force.lvm',
      'SEC10_30CM_N22/force.lvm',
      'SEC9_30CM_N23/force.lvm',
      'SEC8_27.5CM_N24/force.lvm',
      'SEC7_27.5CM_N25/force.lvm',
      'SEC6_25CM_N26/force.lvm',
      'SEC5_25CM_N27/force.lvm',
      'SEC4_25CM_N28/force.lvm',
      'SEC3_22.5CM_N29/force.lvm',
      'SEC2_25CM_N30/force.lvm',
      'SEC1_25CM_N31/cap.lvm']


SPEED=np.asarray(['25600.00',
                  '27648.00',
                  '26404.00',
                  '28928.00',
                  '28928.00',
                  '35328.00',
                  '36608.00',
                  '40704.00',
                  '43264.00',
                  '27904.00',
                  '0023552.0000',
                  '26368.00',
                  '29952.00',
                  '19712.00',
                  '26368.00',
                  '21504.00',
                  '19840.00',
                  '19968.00',
                  '19840.00',
                  '19968.00',#8
                  '20160.00',#7
                  '20328.00',#6
                  '20224.00',#5
                  '21248.00',#4
                  '23040.00',#3
                  '22784.00',
                  '22784.00'],dtype='float')


SPEED= (36.33/1000)/(0.000001*SPEED)
EN=0.5*MASSIMP*(SPEED**2)
#%% SEC data processing

ra= np.arange(0,len(SEC))
p=np.asarray([72.9,
              123.22,
              98.2,
              132,
              129.3,
              111.28,
              195,
              97.5,
              167.5,
              168.5,
              126.2,
              120.2,
              181.2,
              140,
              113.3,
              251.5,
              135.4,
              168.5,
              131.2,
              183.5,
              163.5,
              143.5,
              158.2,
              139.42,
              134.24,
              151.5,
              152.1])

CD=pd.DataFrame()
fig=plt.figure(figsize=(10,4))
for file in ra:    
    D=np.genfromtxt(SEC[file],skip_header=22,delimiter='\t')
    du=np.asarray(D[:,1],dtype=float)
    ttu=np.absolute(np.asarray(D[:,0],dtype=float))
    if ttu[-1] > 2000:
        ttus=272.6932/len(ttu)
        ttu=np.arange(0,272.6932,ttus)
        
    
    # plot
    
    # fig=plt.figure(figsize=(10,4))
    # plt.plot(ttu,du/du[0],"-",label="Capacitance file: "+str(SEC[file]).strip('/cap.lvm'))
    # plt.xlabel('time')
    # plt.ylabel(yl)
    # plt.legend()
    # plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0))
    # plt.tight_layout()
    # plt.savefig('Cap_plot_veiw'+str(file)+'.pdf')

    
    
    # points
    p1=p[file]-30; p2=p[file]-5;
    a1=p[file]+5; a2=p[file]+30;
    
    
    # pre impact
    cc1,tt1=strip_plot(du,ttu,p1,p2)
    ccm1,ccb1=np.polyfit(tt1,cc1,1)
    cclinfac1=ccm1*tt1
    cc1nd=cc1#-cclinfac1 #remove drift
    
    
    # after impact
    cc2,tt2=strip_plot(du,ttu,a1,a2)
    ccm2,ccb2=np.polyfit(tt2,cc2,1)
    cclinfac2=ccm2*tt2
    cc2nd=cc2#-cclinfac2 #remove drift
    tt2=tt2+(a1-p1) #time displacement
    
     
    # region
    cc,tt=strip_plot(du,ttu,p1,a2)
    ccm,ccb=np.polyfit(tt,cc,1)
    cclinfac=np.average([ccm1,ccm2])
    ccnd=cc-cclinfac*tt #remove drift
    cc2nd=cc2-cclinfac*tt2
    cc1nd=cc1-cclinfac*tt1
    
    #plot
    
    fig=plt.figure(figsize=(6,4))
    plt.plot(tt,cc/cc[0],"-")#label="Capacitance file: "+str(SEC[file]).strip('/cap.lvm'))
    plt.xlabel('time')
    plt.ylabel(yl)
    plt.title('impact energy: '+str(round(EN[file],2))+' J')
    # plt.legend()
    plt.ylim(.99,1.010)
    plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0))
    plt.tight_layout()
    plt.savefig('Cap_plot_veiw'+str(file)+'.png',dpi=400)
    
    
    lbtt='tt'#+str(file)
    lbcc='cc'#+str(file)
    
    lbtt1='tt1'#+str(file)
    lbcc1='cc1'#+str(file)
    
    lbtt2='tt2'#+str(file)
    lbcc2='cc2'#+str(file)
    
    
    cav1=np.average(cc1nd)
    cav2=np.average(cc2nd)
    
    delc = (cav1-cav2)/cav1
    C.append(delc)
    
    
    Dict={}
    Dict[lbtt]=list(tt)
    Dict[lbcc]=list(ccnd)
    Dict[lbtt1]=list(tt1)
    Dict[lbcc1]=list(cc1nd)
    Dict[lbtt2]=list(tt2)
    Dict[lbcc2]=list(cc2nd)
        
    
    CD = CD.append(Dict, ignore_index=1)
    

# fig, axs = plt.subplots(nrows=3, ncols=len(ra),figsize=(6.5,7))
# for i in ra:
#     cc=CD.iloc[i,0];  cc1=CD.iloc[i,1]; cc2=CD.iloc[i,2];
#     tt=CD.iloc[i,3];  tt= np.linspace(tt[0],tt[-1],num=len(cc));
#     tt1=CD.iloc[i,4]; tt1=np.linspace(tt1[0],tt1[-1],num=len(cc1));
#     tt2=CD.iloc[i,5]; tt2=np.linspace(tt2[0],tt2[-1],num=len(cc2));
    
#     cav1=np.average(cc1)*np.ones(np.shape(cc1)); 
#     cav2=np.average(cc2)*np.ones(np.shape(cc2));
#     dcav2=((np.average(cc2)-np.average(cc1))/np.average(cc1))*np.ones(np.shape(cc2))
    
#     axs[0,i].plot(tt,cc,"-"); axs[0,i].axis(ymin=0.999*np.min(cc1),ymax=1.001*np.max(cc2));
#     axs[1,i].plot(tt1,cc1,"-",c='tab:green'); axs[1,i].plot(tt2,cc2,"-",c='tab:orange');
#     axs[2,i].plot(tt1,np.zeros(np.shape(cc1)),"-",c='tab:green'); axs[2,i].plot(tt2,dcav2,"-",c='tab:orange');
#     axs[2,i].axis(ymin=-1/1000,ymax=5/1000)
# fig.text(.5,.03,'time (s)', rotation=0)
# fig.text(.015,.57,yl1, rotation=90)
# fig.text(.015,.1,yl, rotation=90)
# fig.tight_layout(rect=[.03, 0.03, 1, 1])   
# fig.savefig('SECdata.pdf')

# fig, axs = plt.subplots(nrows=2, ncols=len(ra),figsize=(10,4))
# for i in ra:
#     cc=CD.iloc[i,0];  cc1=CD.iloc[i,1]; cc2=CD.iloc[i,2];
#     tt=CD.iloc[i,3];  tt= np.linspace(tt[0],tt[-1],num=len(cc));
#     tt1=CD.iloc[i,4]; tt1=np.linspace(tt1[0],tt1[-1],num=len(cc1));
#     tt2=CD.iloc[i,5]; tt2=np.linspace(tt2[0],tt2[-1],num=len(cc2));
    
#     cav1=np.average(cc1)*np.ones(np.shape(cc1)); 
#     cav2=np.average(cc2)*np.ones(np.shape(cc2));
#     dcav2=((np.average(cc2)-np.average(cc1))/np.average(cc1))*np.ones(np.shape(cc2))
    
    
#     axs[0,i].plot(tt,cc,"-"); axs[0,i].axis(ymin=0.999*np.min(cc1),ymax=1.001*np.max(cc2));
#     axs[0,i].plot(tt1,cc1,"-",c='tab:green'); axs[0,i].plot(tt2,cc2,"-",c='tab:orange');
#     axs[1,i].plot(tt1,np.zeros(np.shape(cc1)),"-",c='tab:green'); axs[1,i].plot(tt2,dcav2,"-",c='tab:orange');
#     axs[1,i].axis(ymin=-1/1000,ymax=5/1000)
# fig.text(.5,.03,'time (s)', rotation=0)
# fig.text(.015,.66,yl1, rotation=90)
# fig.text(.015,.2,yl, rotation=90)
# fig.tight_layout(rect=[.03, 0.03, 1, 1])   
# fig.savefig('SECdata1.pdf')

   
C=np.abs(C)

#%%




#%% Load cell data processing

# met=1.4881693
# mass=6.4636

# ED=pd.DataFrame()

# e1=np.asarray([[66.3309,66.3371,66.348],[14.726,14.7317,14.7416],[13.588,13.5946,13.607],[19.278,19.2855,19.2939],[92.917,92.924,92.9337]])
# E=[]
# E1=[]
# E0=[]
# ra= np.arange(0,len(LOAD))
# for file in ra:
#     # Name="plot of %s"%(LOAD[file])
#     # name=LOAD[file].strip('.lvm')
#     # name=name.strip('Data/impact')
#     # N.append(name)
    
#     D=np.genfromtxt(LOAD[file],skip_header=24,delimiter='\t')

#     acc = D[:,1]
#     tt = D[:,0]
    
#     intt1=[]
#     inforce1=[]
#     for i in np.arange(0,len(tt)):
#         if tt[i] > e1[file,1] and tt[i]<e1[file,2]:
#             intt1=np.append(intt1,tt[i])
#             inforce1=np.append(inforce1,acc[i])
#     intt1=intt1-intt1[0]
#     inforce1=inforce1-acc[0]
    
#     delp=np.trapz(inforce1,intt1)
#     impE=(mass/2)*((delp*met)/(2*mass))**2 
#     E1=np.append(E1,impE)
    
#     intt0=[]
#     inforce0=[]
#     for i in np.arange(0,len(tt)):
#         if tt[i] > e1[file,0] and tt[i]<e1[file,1]:
#             intt0=np.append(intt0,tt[i])
#             inforce0=np.append(inforce0,acc[i])
#     intt0=intt0-intt1[0]
#     inforce0=inforce0-acc[0]
    
#     delp=np.trapz(inforce0,intt0)
#     impE=(mass/2)*((delp*met)/(2*mass))**2
#     E0=np.append(E0,impE)
    
    
#     intt=[]
#     inforce=[]
#     for i in np.arange(0,len(tt)):
#         if tt[i] > e1[file,0] and tt[i]<e1[file,2]:
#             intt=np.append(intt,tt[i])
#             inforce=np.append(inforce,acc[i])
#     intt=intt-intt[0]
#     inforce=inforce-acc[0]
#     # intt=[]
#     # inforce=[]
#     # for i in np.arange(0,len(tt)):
#     #     if tt[i] > e1[file,0] and tt[i]<tt[np.where(acc==np.max(acc))]:
#     #         intt=np.append(intt,tt[i])
#     #         inforce=np.append(inforce,acc[i])
#     # intt=intt-intt[0]
#     # inforce=inforce-acc[0]
    
#     delp=np.trapz(inforce,intt)
        
#     #impE=6.5*(delp/2*.3048)**2
#     impE=(mass/2)*((delp*met)/(2*mass))**2 -.093
#     E=np.append(E,impE)
    
#     lbtt='tt'#+str(file)
#     lbinf='inf'#+str(file)
    
#     lbtt0='tt0'#+str(file)
#     lbinf0='inf0'#+str(file)
    
#     lbtt1='tt1'#+str(file)
#     lbinf1='inf1'#+str(file)
    
#     Dict={}
#     Dict[lbtt]=list(intt)
#     Dict[lbinf]=list(0.00444822162*inforce)
#     Dict[lbtt0]=list(intt0)
#     Dict[lbinf0]=list(0.00444822162*inforce0)
#     Dict[lbtt1]=list(intt1)
#     Dict[lbinf1]=list(0.00444822162*inforce1)
        
    
#     ED = ED.append(Dict, ignore_index=1)
    
    
#     # inforce=moving_average(inforce, 2)
#     # inforce1=moving_average(inforce1, 2)
#     # fig, (ax1, ax2) = plt.subplots(2)
#     # fig.suptitle(LOAD[file])
#     # ax1.plot(np.linspace(tt[0],tt[-1],num=len(acc)),acc,"-")
#     # ax2.plot(np.linspace(intt0[-1],intt1[-1],num=len(inforce1)),inforce1,"-",c='tab:green')
#     # ax2.plot(np.linspace(intt0[0],intt0[-1],num=len(inforce0)),inforce0,"-",c='tab:red')

# # fig, axs = plt.subplots(nrows=1, ncols=len(ra),figsize=(10,2))
# # for i in ra:
# #     inf=ED.iloc[i,0];  inf1=ED.iloc[i,1]; inf2=ED.iloc[i,2];
# #     tt=ED.iloc[i,3];  tt= np.linspace(tt[0],tt[-1],num=len(inf))-tt[0];
# #     tt1=ED.iloc[i,4]; tt1=np.linspace(tt1[0],tt1[-1],num=len(inf1))-tt1[0];
# #     tt2=ED.iloc[i,5]; tt2=np.linspace(tt2[0],tt2[-1],num=len(inf2))-tt2[0];
    
# #     bo1=inf[0]*np.ones(np.shape(inf1)); 
    
# #     axs[i].plot(tt,inf,"-");
# #     axs[i].set_title(str(str(round(E[i],3))+'J'))
# #     axs[i].axis(ymin=0.95*np.min(inf),ymax=1.05*np.max(inf));
# #     axs[i].plot(tt1,inf1,"-",c='tab:orange'); 
# #     axs[i].fill_between(tt1, bo1, inf1, color="tab:orange", alpha=.5)
# #     axs[i].scatter(tt1[-1],inf1[-1],color="tab:orange") 
# # fig.text(.5,.03,'time (s)', rotation=0)
# # fig.text(.015,.45,'force (kN)', rotation=90)
# # fig.tight_layout(rect=[.03, 0.03, 1, 1])   
# # fig.savefig('loadcelldata.pdf')    



# # L= np.asarray([5,10,15,20,25]) 
# # PE=mass*9.81*L/100

#%% PLOT view

fig=plt.figure(figsize=(6,4))

plt.plot(EN,C,'o',c='tab:green',label="cap")
#plt.contourf(x0,x1,zp,cmap='Pastel2')
#contour=plt.contour(x0,x1,zp,[0.6779])
#plt.scatter(E, C, c='k' , marker='o', label="impact trial")
plt.xlabel(xl)
plt.ylabel(yl)
plt.ylim(0,np.max(C)*1.15)
#plt.xlim(0,np.max(E)*1.15)
#plt.text(0.6779-.05,.0012,'nominal proof resilliance', rotation=90)
plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0))
plt.tight_layout()
plt.savefig('test_veiw.png',dpi=400)

#%% plot all


# E=E0

ent=5.084321175
ena=2.881115333
enb=0.67790949



sort_list = pd.DataFrame.from_dict({'energy':EN,'cap':C})
sort_list = sort_list.sort_values("energy")
arr = sort_list.values

Es=arr[:,0]; Cs=arr[:,1]

fig=plt.figure(figsize=(6,4))
plt.plot(Es,Cs,'o',c='tab:green',label="cap")
plt.xlabel(xl)
plt.ylabel(yl)
plt.ylim(0,np.max(C)*1.15)
plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0))
plt.tight_layout()

Est= np.where(Es>ent,Es,np.nan)
Est= Est[~np.isnan(Est)]
Cst= np.where(Es>ent,Cs,np.nan)
Cst= Cst[~np.isnan(Cst)]

Esb= np.where(Es<ent,Es,np.nan)
Esb= Esb[~np.isnan(Esb)]
Csb= np.where(Es<ent,Cs,np.nan)
Csb= Csb[~np.isnan(Csb)]


x0, x1 = np.meshgrid(np.linspace(0,Es[-1]+1,800),np.linspace(0,1,2))

k=.4
b=5
Cbre= 1/(1+np.exp(-k*(x0-b)))

m1, b1 = np.polyfit(Esb, Csb, 1)
m2, b2 = np.polyfit(Est, Cst, 1)

Xm = np.linspace(0,Es[-1]+1,num=200);
yb = (0*Xm + b1); yt = (m2*Xm + b2);
Ym=np.where(yb>yt,yb,yt);
    
# Ym=[]
# for x in Xm:
#     if (m1*x + b1) > (m2*x + b2):
#         Ym= np.append(Ym,(m1*x + b1))
#     else:
#         Ym= np.append(Ym,(m2*x + b2))


fig=plt.figure(figsize=(10,4))
plt.plot(Xm,Ym,"k--",label="trend")
plt.plot([ent,ent],[0,1],"k-")
plt.contourf(x0,x1,Cbre,cmap='summer',levels=800,vmin=0,vmax=1.2)
plt.scatter(Es, Cs, c='k' , marker='o', label="impact trial")
plt.xlabel(xl)
plt.ylabel(yl)
plt.ylim(0,np.max(Cs)*1.15)
plt.xlim(2,np.max(x0))
plt.legend()
plt.text(ent-.18,.0012,'nominal proof resilliance', rotation=90)
plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0))
plt.tight_layout()
plt.savefig('fitcap.pdf')

fig=plt.figure(figsize=(10,4))
plt.plot(Xm,Ym/4,"k--",label="trend")
plt.plot([ent,ent],[0,1],"k-")
plt.contourf(x0,x1,Cbre,cmap='summer',levels=800,vmin=0,vmax=1.2)
plt.scatter(Es, Cs/4, c='k' , marker='o', label="impact trial")
plt.xlabel(xl)
plt.ylabel('average strain '+ r"$\left( \frac{\mathrm{mm}}{\mathrm{mm}} \right)$")
plt.ylim(0,np.max(Cs/4)*1.15)
plt.xlim(2,np.max(x0))
plt.legend()
plt.text(ent-.18,.0012/4,'nominal proof resilliance', rotation=90)
plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0))
plt.tight_layout()
plt.savefig('fitstrain_large.png',dpi=400,transparent=1)

fig=plt.figure(figsize=(8,4))
plt.plot(Xm,Ym/4,"k--",label="trend")
plt.plot([ent,ent],[0,1],"k-")
plt.contourf(x0,x1,Cbre,cmap='summer',levels=800,vmin=0,vmax=1.2)
plt.scatter(Es, Cs/4, c='k' , marker='o', label="impact trial")
plt.xlabel(xl)
plt.ylabel('average strain '+ r"$\left( \frac{\mathrm{mm}}{\mathrm{mm}} \right)$")
plt.ylim(0,np.max(Cs/4)*1.15)
plt.xlim(2,np.max(x0))
plt.legend()
plt.text(ent-.22,.0012/4,'nominal proof resilliance', rotation=90)
plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0))
plt.tight_layout()
plt.savefig('fitstrain_med.png',dpi=400,transparent=1)

fig=plt.figure(figsize=(6,4))
plt.plot(Xm,Ym/4,"k--",label="trend")
plt.plot([ent,ent],[0,1],"k-")
plt.contourf(x0,x1,Cbre,cmap='summer',levels=800,vmin=0,vmax=1.2)
plt.scatter(Es, Cs/4, c='k' , marker='o', label="impact trial")
plt.xlabel(xl)
plt.ylabel('average strain '+ r"$\left( \frac{\mathrm{mm}}{\mathrm{mm}} \right)$")
plt.ylim(0,np.max(Cs/4)*1.15)
plt.xlim(2,np.max(x0))
plt.legend()
plt.text(ent-.28,.0012/4,'nominal proof resilliance', rotation=90)
plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0))
plt.tight_layout()
plt.savefig('fitstrain_small.png',dpi=400,transparent=1)
#%% plot all

# x0, x1 = np.meshgrid(np.linspace(0,4,800),np.linspace(0,1,2))
# # E=E0

# m1, b1 = np.polyfit([E[0],E[1],E[2]], [C[0],C[1],C[2]], 1)
# m2, b2 = np.polyfit([E[2],E[3],E[4]], [C[2],C[3],C[4]], 1)

# Xm = np.linspace(0,E[-1]+1,num=200)
# Ym=[]
# for x in Xm:
#     if (m1*x + b1) > (m2*x + b2):
#         Ym= np.append(Ym,(m1*x + b1))
#     else:
#         Ym= np.append(Ym,(m2*x + b2))

# print(m2)

# zp = 0*x0
# for i in np.arange(0,2):
#     for j in np.arange(0,600):
#         if x0[i,j] > 0.6779 and x0[i,j]<4.2369:
#             zp[i,j]=1
#         elif x0[i,j] >= 4.2369:
#             zp[i,j]=0
#         else:
#             zp[i,j]=0

# model =[]
# k=(1-.3**2)/((4.3e+10)*(0.003175**3))
# modela = 10.4*k*(Xm-0.6779)
# for i in modela:
#     if i < 0:
#         model=np.append(model,np.nan)
#     else:
#         model=np.append(model,i)


# GFcorr=5.364658405646891

# Xmex = np.linspace(0.678,E[-1]+1,num=200)
# modelexp =[]
# Eyoung=4.3e+10; V=0.0154838*0.003175; K=2/(Eyoung*V);
# K=2/((Eyoung*(V))/(8*(1-.21**2)))
# modela = 2*(np.exp((K*(Xmex-0.6779))**.5)-1)
# for i in modela:
#     if i < 0:
#         modelexp=np.append(modelexp,0)
#     else:
#         modelexp=np.append(modelexp,i)
     

# modelexfit=[]
# mmexfit,bbexfit=np.polyfit(Xmex,modelexp,1)
# modela = mmexfit*(Xmex-0.6779) - bbexfit
# for i in modela:
#     if i < 0:
#         modelexfit=np.append(modelexfit,np.nan)
#     else:
#         modelexfit=np.append(modelexfit,i)
        
    

# fig=plt.figure(figsize=(10,4))

# plt.plot(Xm,Ym,"k--",label="trend")
# #plt.plot(Xm,model,"--",c='tab:gray',label="model GF 5.2")
# plt.plot(Xmex,modelexfit,"--",c='tab:green',label="model exp fit")
# plt.plot(Xmex,modelexp,"-",c='tab:green',label="model exp")
# plt.contourf(x0,x1,zp,cmap='Pastel2')
# contour=plt.contour(x0,x1,zp,[0.6779])
# plt.scatter(E, C, c='k' , marker='o', label="impact trial")
# plt.xlabel(xl)
# plt.ylabel(yl)
# plt.ylim(0,np.max(C)*1.15)
# plt.xlim(0,np.max(E)*1.15)
# plt.legend()
# plt.text(0.6779-.05,.0012,'nominal proof resilliance', rotation=90)
# plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0))
# plt.tight_layout()
# plt.savefig('fit5.pdf')

#%%





# fig, axs = plt.subplots(nrows=3, ncols=len(ra),figsize=(10,6))
# for i in ra:
#     cc=CD.iloc[i,0];  cc1=CD.iloc[i,1]; cc2=CD.iloc[i,2];
#     tt=CD.iloc[i,3];  tt= np.linspace(tt[0],tt[-1],num=len(cc));
#     tt1=CD.iloc[i,4]; tt1=np.linspace(tt1[0],tt1[-1],num=len(cc1));
#     tt2=CD.iloc[i,5]; tt2=np.linspace(tt2[0],tt2[-1],num=len(cc2));
    
#     cav1=np.average(cc1)*np.ones(np.shape(cc1)); 
#     cav2=np.average(cc2)*np.ones(np.shape(cc2));
#     dcav2=((np.average(cc2)-np.average(cc1))/np.average(cc1))*np.ones(np.shape(cc2))
    
    
#     axs[1,i].plot(tt,cc,"-"); axs[1,i].axis(ymin=0.999*np.min(cc1),ymax=1.001*np.max(cc2));
#     axs[1,i].plot(tt1,cc1,"-",c='tab:green'); axs[1,i].plot(tt2,cc2,"-",c='tab:orange');
#     axs[2,i].plot(tt1,np.zeros(np.shape(cc1)),"-",c='tab:green'); axs[2,i].plot(tt2,dcav2,"-",c='tab:orange');
#     axs[2,i].axis(ymin=-1/1000,ymax=5/1000)
    
#     inf=ED.iloc[i,0];  inf1=ED.iloc[i,1]; inf2=ED.iloc[i,2];
#     ttf=ED.iloc[i,3];  ttf= np.linspace(ttf[0],ttf[-1],num=len(inf))-ttf[0];
#     ttf1=ED.iloc[i,4]; ttf1=np.linspace(ttf1[0],ttf1[-1],num=len(inf1))-ttf1[0];
#     ttf2=ED.iloc[i,5]; ttf2=np.linspace(ttf2[0],ttf2[-1],num=len(inf2))-ttf2[0];
    
#     bo1=inf[0]*np.ones(np.shape(inf1)); 
    
#     axs[0,i].plot(ttf,inf,"-");
#     axs[0,i].set_title(str(str(round(E[i],3))+'J'))
#     axs[0,i].axis(ymin=0.95*np.min(inf),ymax=1.05*np.max(inf));
#     axs[0,i].plot(ttf1,inf1,"-",c='tab:orange'); 
#     axs[0,i].fill_between(ttf1, bo1, inf1, color="tab:orange", alpha=.5)
#     axs[0,i].scatter(ttf1[-1],inf1[-1],color="tab:orange")
# fig.text(.016,.75,'force (kN)', rotation=90) 
# fig.text(.5,.03,'time (s)', rotation=0)
# fig.text(.013,.45,yl1, rotation=90)
# fig.text(.013,.15,yl, rotation=90)
# fig.tight_layout(rect=[.03, 0.03, 1, 1])   
# fig.savefig('SECloadcom.pdf')