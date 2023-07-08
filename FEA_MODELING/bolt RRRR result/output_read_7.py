# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 20:18:47 2020

@author: alexv
"""


# import IPython as IP
# IP.get_ipython().magic('reset -sf')

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import re
import pandas as pd
from scipy import fftpack, signal # have to add 
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import linear_model
from sklearn import pipeline
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp2d
import scipy.ndimage as ndi
import time as time

from matplotlib import rc
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

time1=time.time()
def surfaceIntegrationPlane(X,Y,Z):
    atotal=0
    for index, _ in np.ndenumerate(X):
        xi,yi = index
        if np.max([xi,yi]) >= len(X.ravel())**.5-1:
            # print('out of range')
            None
        else:
            #print('in range')
            x0, y0, x1, y1 = X[xi,yi], Y[xi,yi], X[xi,yi+1], Y[xi+1,yi]
            dy, dx = abs(y0-y1), abs(x0-x1)
            z00, z10, z01, z11 = Z[xi,yi], Z[xi+1,yi], Z[xi,yi+1], Z[xi+1,yi+1]
            # fore triangle
            foreL10 = [z10-z00,dx,0]
            foreL01 = [z01-z00,0,dy]
            area_fore = abs(np.linalg.norm(np.cross(foreL10,foreL01))/2)
            # aft triangle
            aftL10 = [z10-z11,dx,0]
            aftL01 = [z01-z11,0,dy]
            area_aft = abs(np.linalg.norm(np.cross(aftL10,aftL01))/2)
            # area accumulator
            atotal += area_fore + area_aft
    return atotal


def slopefit(x,y):
    x,y=np.array(x),np.array(y)
    mse_min = float('inf')
    for i in np.linspace(0,1,1000):
        mse = np.sum(np.square(np.subtract(y,i*x)))
        if mse < mse_min:
            factor , mse_min = i, mse
    return factor
    
def plot_surf(array):
    # Max Min
    dmax=np.nanmax(array)
    dmin=np.nanmin(array)

    # get array shape
    Dshape=np.shape(array)
    
    # get array of x and y
    x0, x1 = np.meshgrid(np.linspace(0,1,Dshape[1]),np.linspace(0,1,Dshape[0]))
    
    # Open figure
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})  
    
    # Plot surface
    surf = ax.plot_surface(x0, x1, array, cmap=cm.viridis,linewidth=0, antialiased=1 ,vmax=dmax,vmin=dmin)
    
    # Customize the z axis.
    #ax.set_zlim(dmin-0.1*dmin, dmax+0.1*dmax)
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    return fig
 
plt.close('all')

#%% Load data

N = np.loadtxt('coord.txt',dtype=float,delimiter=',')
F_name = ['Step-01_results.txt','Step-02_results.txt','Step-03_results.txt','Step-1_results.txt',
          'Step-11_results.txt','Step-12_results.txt','Step-13_results.txt','Step-2_results.txt',
          'Step-21_results.txt','Step-22_results.txt','Step-23_results.txt','Step-3_results.txt']

DELCN=[]
DELSEC=[]

string_filter = ['[ ',' ]','[',']']

#trial 1
#base  = (1.7942057690769743E-10 + 1.7944074181939352E-10)/2
#smallest = (1.7961500596467843E-10 + 1.7962619663445521E-10)/2
#medium = (1.7981823275574808E-10 + 1.7983396771076307E-10)/2
#biggest = (1.8028108717094319E-10 + 1.802248754081973E-10)/2
#dcoc=[(smallest-base)/base,(medium-base)/base,(biggest-base)/base]

# trial2
# ExData={'baseline':[[1.7982716561146276E-10,1.7983169443518835E-10],[1.7983260196601138E-10,1.7983141262912369E-10],[1.7983167474175264E-10,1.7983637890703088E-10]],
#         'disp0000':[[1.7972395138287235E-10,1.7980938563812062E-10],[1.7973618117294240E-10,1.7972277837387531E-10],[1.7970801586137945E-10,1.7971958973675451E-10]],
#         'disp0250':[[1.7973014848383869E-10,1.7987472959013668E-10],[1.7972840986337896E-10,1.7987316581139623E-10],[1.7974352022659104E-10,1.7996677050983012E-10]],
#         'disp0500':[[1.7976909503498836E-10,1.8027579143618806E-10],[1.7988629253582143E-10,1.8028565964678448E-10],[1.7992209050316558E-10,1.8032408617127620E-10]],
#         'disp0750':[[1.7996305798067307E-10,1.8072573388870376E-10],[1.7997409930023329E-10,1.8067932472509175E-10],[1.7992832639120294E-10,1.8071628267244256E-10]],
#         'disp1000':[[1.7997398790403208E-10,1.8112381896034652E-10],[1.7980890549816725E-10,1.8093277054315225E-10],[1.7968615108297243E-10,1.8092218663778731E-10]],
#         'disp1250':[[1.7981172935688101E-10,1.8115636334555152E-10],[1.7977761952682446E-10,1.8117114785071646E-10],[1.7975181649450182E-10,1.8116787200933018E-10]],
#         'disp1500':[[1.7992087867377548E-10,1.8183890716427870E-10],[1.7980799706764406E-10,1.8183407380873032E-10],[1.7977735531489505E-10,1.8182172219260244E-10]],
#         'disp1750':[[1.7998023235588149E-10,1.8193259833388864E-10],[1.7989348550483174E-10,1.8223096314561819E-10],[1.7987334508497167E-10,1.8221498953682097E-10]],
#         'disp2000':[[1.7991881989336887E-10,1.8230909730089959E-10],[1.7978331799400192E-10,1.8230530943018990E-10],[1.7984956237920691E-10,1.8229852895701437E-10]],
#         'disp2250':[[1.7975293968677115E-10,1.8270148510496508E-10],[1.7970549716761088E-10,1.8269755451516159E-10],[1.7968700176607807E-10,1.8268558510496490E-10]],
#         'disp2500':[[1.7970442612462512E-10,1.8343081749416864E-10],[1.7974852789070313E-10,1.8340537934022002E-10],[1.7980132659113622E-10,1.8339128933688764E-10]],
#         'disp2750':[[1.7995867397534155E-10,1.8356973365544816E-10],[1.7979781732755751E-10,1.8358959603465505E-10],[1.7984500096634456E-10,1.8361029596801069E-10]],
#         'disp3000':[[1.7973891696101304E-10,1.8385656091302906E-10],[1.7969367270909686E-10,1.8392093372209252E-10],[1.7971164281906023E-10,1.8388048177274241E-10]]}


base=1.7929210159946679e-10
ExData={'baseline':[[base,1.7929099953348874E-10], [base,1.7928957257580804E-10], [base,1.7929068557147619E-10], [base,1.7929714871709422E-10]],
        'disp0000':[[base,1.7929714871709422E-10], [base,1.7928048090636454E-10], [base,1.7923740023325549E-10], [base,1.7923563595468175E-10]],
        'disp0250':[[base,1.7981293972009324E-10], [base,1.7979574921692769E-10], [base,1.7977466181272903E-10], [base,1.7974456787737421E-10]],
        'disp0500':[[base,1.8044236931022981E-10], [base,1.8056648447184265E-10], [base,1.8055465041652790E-10], [base,1.8056206164611798E-10]],
        'disp0750':[[base,1.8136890803065633E-10], [base,1.8141429446851045E-10], [base,1.8138969286904369E-10], [base,1.8138236121292897E-10]],
        'disp1000':[[base,1.8195644271909366E-10], [base,1.8195455281572803E-10], [base,1.8195008540486512E-10], [base,1.8195906261246254E-10]],
        'disp1250':[[base,1.8282132869043658E-10], [base,1.8290877304231923E-10], [base,1.8291627864045323E-10], [base,1.8291790359880045E-10]],
        'disp1500':[[base,1.8352008280573152E-10], [base,1.8362362305898013E-10], [base,1.8364315584805068E-10], [base,1.8367325468177273E-10]],
        'disp1750':[[base,1.8460492682439185E-10], [base,1.8467789183605461E-10], [base,1.8468247494168595E-10], [base,1.8467862525824721E-10]],
        'disp2000':[[base,1.8530389606797731E-10], [base,1.8536649330223258E-10], [base,1.8536144065311562E-10], [base,1.8538411912695761E-10]],
        'disp2250':[[base,1.8612723558813727E-10], [base,1.8623015301566149E-10], [base,1.8622288607130954E-10], [base,1.8623782285904704E-10]],
        'disp2500':[[base,1.8711542469176948E-10], [base,1.8713098237254262E-10], [base,1.8713193445518162E-10], [base,1.8714151799400206E-10]],
        'disp2750':[[base,1.8777712192602459E-10], [base,1.8767746557814062E-10], [base,1.8767369353548823E-10], [base,1.8776615214928350E-10]],
        'disp3000':[[base,1.8912925038320557E-10], [base,1.8894595538153946E-10], [base,1.8912105248250590E-10], [base,1.8910994825058302E-10]]}


dcoc1 = []
dcoc2 = []
dcoc3 = []
for key, trials in ExData.items():
    t1 , t2, t3 , t4= trials
    dcoc1.append((t1[1]-t1[0])/t1[0])
    dcoc2.append((t2[1]-t2[0])/t2[0])
    dcoc3.append((t3[1]-t3[0])/t3[0])
    
cap_sample =  np.loadtxt('cap.lvm',dtype=float,delimiter='\t')
cap_sample = [ c for t , c in cap_sample if t < 40 ]
cap_std = np.std(cap_sample)
sample_pop = len(cap_sample)
cap_ci_98 = 2.33*cap_std/(sample_pop**0.5) 
  
abadcoc={}
deform={}
undeform={}
for file in np.arange(0,len(F_name)): 
    dis = np.loadtxt(F_name[file],dtype=str,delimiter='\t',skiprows=1)
    Dis = np.transpose(np.expand_dims(dis[:,0],axis=1))
    # num = dis[:,0]  
    # dis = dis[:,1]
    D = [[0,0,0]]
    node_call=set(dis[:,0])
    lnum=0
    for num, entry in dis:
        if(num in node_call)and(lnum<int(num)):
            lnum=int(num)
            node_call.remove(num)
            line = entry
            for i in range(0,19):
                line = line.replace(' '*(20-i),' ')
            for s in string_filter:
                line = line.replace(s,'')
            line = np.expand_dims(np.array(line.split(' '),dtype=float),axis=0)
            D = np.append(D,line,axis=0)
    D = np.delete(D,0,0)
    D = np.insert(D,0,np.zeros(np.shape(N)[0]),axis=1)
    del(dis);del(Dis);del(entry);del(line);
    
    F = D+N
    
    Original  = {'Global Node Number' : N[:,0],
                'X_coord' : N[:,1],
                'Y_coord' : N[:,2],
                'Z_coord' : N[:,3],
                }
    Displace = {'Global Node Number' : F[:,0],
                'X_coord' : F[:,1],
                'Y_coord' : F[:,2],
                'Z_coord' : F[:,3],
                }            
        
    region = 0.501
    NOI = set()
    for index, node in enumerate(Original['Global Node Number']):
        if abs(float(Original['X_coord'][index]))<region and abs(float(Original['Y_coord'][index]))<region and abs(float(Original['Z_coord'][index])) > 0.124:
            NOI.add(int(node))
            
    SEC_undeform = np.array([[row[0],row[1],row[2],row[3]] for row in N if row[0] in NOI])
    SEC_deform = np.array([[row[0],row[1],row[2],row[3]] for row in F if row[0] in NOI])
    
    SEC_undeform_plane = np.ones((int(len(NOI)**.5),int(len(NOI)**.5)))
    SEC_undeform_planeX = np.ones((int(len(NOI)**.5),int(len(NOI)**.5)))
    SEC_undeform_planeY = np.ones((int(len(NOI)**.5),int(len(NOI)**.5)))
    
    SEC_deform_plane = np.ones((int(len(NOI)**.5),int(len(NOI)**.5)))
    SEC_deform_planeX = np.ones((int(len(NOI)**.5),int(len(NOI)**.5)))
    SEC_deform_planeY = np.ones((int(len(NOI)**.5),int(len(NOI)**.5)))
    
    for ini, i in enumerate(np.unique(np.round(SEC_undeform[:,1],2))):
        for inj, j in enumerate(np.unique(np.round(SEC_undeform[:,2],2))):
            for row in SEC_undeform:
                if i==np.round(row[1],2) and j==np.round(row[2],2):
                    SEC_deform_plane[ini,inj] = row[0]
                    SEC_undeform_planeX[ini,inj]=row[2]
                    SEC_undeform_planeY[ini,inj]=row[1]
                    SEC_undeform_plane[ini,inj] = row[3]
    
    SEC_undeform_sorted=np.array([list(NOI),SEC_undeform_planeX.ravel(),
                                  SEC_undeform_planeY.ravel(),
                                  SEC_undeform_plane.ravel()]).T
    
    for ind in np.ndindex((int(len(NOI)**.5),int(len(NOI)**.5))):
        for row in SEC_deform:
            if SEC_deform_plane[ind]==row[0]:
                SEC_deform_planeX[ind]=row[2]
                SEC_deform_planeY[ind]=row[1]
                SEC_deform_plane[ind]=row[3]
                
    SEC_deform_sorted=np.array([list(NOI),SEC_deform_planeX.ravel(),
                                SEC_deform_planeY.ravel(),
                                SEC_deform_plane.ravel()]).T
    
                
    undeform.update({F_name[file]:{'Z':SEC_undeform_plane,'X':SEC_undeform_planeX,'Y':SEC_undeform_planeY}})
    deform.update({F_name[file]:{'Z':SEC_deform_plane,'X':SEC_deform_planeX,'Y':SEC_deform_planeY}})
    #print(np.size(SEC_deform_plane))
    
    undeformedArea=surfaceIntegrationPlane(SEC_undeform_planeX,
                                           SEC_undeform_planeY,
                                           SEC_undeform_plane)
    
    deformedArea=surfaceIntegrationPlane(SEC_deform_planeX,
                                         SEC_deform_planeY,
                                         SEC_deform_plane)
    
    

    
    # plot_surf(SEC_undeform_plane)
    # plot_surf(SEC_deform_plane)
    
    abadcoc.update({F_name[file]:((deformedArea**2)/(undeformedArea**2)-1)})
    
    print(F_name[file])
    print('Predicted change in capacitance from abaqus:')
    print(((deformedArea**2)/(undeformedArea**2)-1)/10)
    print('\n')
    
    
    

f = open("ABAQUS_RES.txt", "w")
f.write(str(abadcoc))
f.close()

t=time.time()-time1
print('time: '+str(t))
xspan=25.4*np.array([0,.025,.05,.075,.1,.125,.15,.175,.2,.225,.25,.275,.3])   
plt.figure()
measure_dcoc1 = dcoc1[1:]
measure_dcoc2 = dcoc2[1:]
measure_dcoc3 = dcoc3[1:]
abaqus_dcoc = np.append([0],[abadcoc[file] for file in F_name])
# abaqus_dcoc_max = np.append([0],[max(abadcoc[file]) for file in F_name])
# abaqus_dcoc_min = np.append([0],[min(abadcoc[file]) for file in F_name])
# xspan_all=[]
# for xpos in xspan:
#     for _ in range(65):
#         xspan_all.append(xpos)
# abaqus_dcoc_var = [0 for _ in range(65)]
# for file in F_name:
#     for dcoc in abadcoc[file]:
#         abaqus_dcoc_var.append(dcoc)
plt.plot(xspan,measure_dcoc1)
plt.plot(xspan,measure_dcoc2)
plt.plot(xspan,abaqus_dcoc)

# m=slopefit(xspan,abaqus_dcoc_min)
# print(m)
# yspan_min=m*xspan

# m=slopefit(xspan,abaqus_dcoc_max)
# print(m)
# yspan_max=m*xspan #+b

m=slopefit(xspan,abaqus_dcoc)
yspan=m*xspan #+b
onedegmse = np.average(np.square(yspan-abaqus_dcoc))
print('Error 1deg: '+str( onedegmse ))
print('\n')

plt.figure()
plt.plot(xspan,abaqus_dcoc)
# plt.plot(xspan,abaqus_dcoc_min)
# plt.plot(xspan,abaqus_dcoc_max)
plt.plot(xspan,yspan)

m =slopefit(xspan,abaqus_dcoc)
yspan= m*xspan


plt.figure()
plt.plot(xspan,abaqus_dcoc)
# plt.plot(xspan,abaqus_dcoc_max )
# plt.plot(xspan,abaqus_dcoc_min )
plt.plot(xspan,yspan)

plt.figure()
plt.plot(xspan,measure_dcoc1)
plt.plot(xspan,measure_dcoc2)
plt.plot(xspan,measure_dcoc3)
# plt.plot(xspan,abaqus_dcoc_max )


exp_mean_dcoc=[np.average([measure_dcoc1[i],measure_dcoc2[i],measure_dcoc3[i]]) for i in range(len(xspan))]
exp_deviant=[np.argmax([(exp_mean_dcoc[i]-measure_dcoc1[i])**2,(exp_mean_dcoc[i]-measure_dcoc2[i])**2,(exp_mean_dcoc[i]-measure_dcoc3[i])**2]) for i in range(len(xspan))]
exp_variant=[[measure_dcoc1[i],measure_dcoc2[i],measure_dcoc3[i]][exp_deviant[i]] for i in range(len(xspan))]
expfitm=slopefit(xspan,exp_mean_dcoc)
exp_yspan=expfitm*xspan

res=[]
for i in range(len(xspan)):
    res=np.append(res,[(measure_dcoc1[i]-exp_yspan[i])**2,(measure_dcoc2[i]-exp_yspan[i])**2, (measure_dcoc3[i]-exp_yspan[i])**2])
    
res_std =((np.sqrt(sum(res)/(12)))**1)/((12)**0.5)
res_ci_98 = 4.5*res_std 


plt.figure()
plt.scatter(xspan,measure_dcoc1)
plt.scatter(xspan,measure_dcoc2)
plt.scatter(xspan,measure_dcoc3)
plt.plot(xspan,exp_yspan+res_ci_98)
plt.plot(xspan,exp_yspan-res_ci_98)
plt.plot(xspan,yspan)

plt.figure()
plt.plot(xspan,exp_yspan)
plt.plot(xspan,yspan)

factor=1
plt.figure(figsize=(8,5))
plt.plot(xspan,factor*exp_yspan,linestyle=':',color='tomato')
plt.scatter(xspan,factor*np.array(measure_dcoc1),color='tab:orange')
plt.scatter(xspan,factor*np.array(measure_dcoc2),color='tab:orange')
plt.scatter(xspan,factor*np.array(measure_dcoc3),color='tab:orange')
plt.plot(xspan,factor*(exp_yspan+res_ci_98), 'g--',label=r'expermental mean $\pm$ C.I. 99.7%')
plt.plot(xspan,factor*(exp_yspan-res_ci_98), 'g--')
#plt.scatter(xspan_all, abaqus_dcoc_var,color='tab:green')
#plt.plot(xspan,yspan,color='steelblue',label='abaqus model')
# plt.plot(xspan,yspan_max,color='steelblue',linestyle=':',label='abaqus model bound')
# plt.plot(xspan,yspan_min,color='steelblue',linestyle=':')
plt.xlabel('deflection by boundary (mm)')
plt.ylabel('capacitance '+r'$(\Delta\%)$')
plt.xlim(0,max(xspan))
plt.ylim(0,max(exp_yspan+res_ci_98)*1.1)
#plt.title('gauge factor of '+str(round(factor,3)))
plt.legend()
plt.tight_layout()
plt.savefig('JigEx_Model.png',dpi=300)

factor=1
r=5/8
plt.figure(figsize=(r*8,r*5))
plt.plot(xspan,factor*exp_yspan,linestyle=':',color='tomato')
plt.scatter(xspan,factor*np.array(measure_dcoc1),color='tab:orange')
plt.scatter(xspan,factor*np.array(measure_dcoc2),color='tab:orange')
plt.scatter(xspan,factor*np.array(measure_dcoc3),color='tab:orange')
plt.plot(xspan,factor*(exp_yspan+res_ci_98), 'g--',label=r'expermental mean $\pm$ C.I. 99.7%')
plt.plot(xspan,factor*(exp_yspan-res_ci_98), 'g--')
#plt.scatter(xspan_all, abaqus_dcoc_var,color='tab:green')
#plt.plot(xspan,yspan,color='steelblue',label='abaqus model')
# plt.plot(xspan,yspan_max,color='steelblue',linestyle=':',label='abaqus model bound')
# plt.plot(xspan,yspan_min,color='steelblue',linestyle=':')
plt.xlabel('deflection by boundary (mm)')
plt.ylabel('capacitance '+r'$(\Delta\%)$')
plt.xlim(0,max(xspan))
plt.ylim(0,max(exp_yspan+res_ci_98)*1.1)
#plt.title('gauge factor of '+str(round(factor,3)))
plt.legend()
plt.tight_layout()
plt.savefig('JigEx_Model_sm.png',dpi=300)


























