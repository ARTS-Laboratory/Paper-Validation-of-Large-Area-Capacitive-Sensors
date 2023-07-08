# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 20:18:47 2020

@author: alexv
"""


import matplotlib.pyplot as plt
plt.close('all')
import IPython as IP
IP.get_ipython().magic('reset -sf')

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


# def Zfunc(X,Y,M):
#     pos=[X,Y]
#     model=mod
#     Z = model.predict(pos)+model._final_estimator.intercept_
#     return Z

def surfaceIntegration(Z,resolution=1000):
    func = interp2d(Z[:,1],Z[:,2],Z[:,3],kind='linear')
    x,dx = np.linspace(min(Z[:,1]),max(Z[:,1]), resolution),(max(Z[:,1])-min(Z[:,1]))/resolution
    y,dy = np.linspace(min(Z[:,2]),max(Z[:,2]), resolution),(max(Z[:,2])-min(Z[:,2]))/resolution
    X, Y = np.meshgrid(x, y)
    plot_surf(func(X.ravel(),Y.ravel()))
    atotal=0
    for index, _ in np.ndenumerate(X):
        xi,yi = index
        if np.max([xi,yi]) >= resolution-1:
            # print('out of range')
            None
        else:
            # print('in range')
            x0, y0, x1, y1 = X[xi,yi], Y[xi,yi], X[xi,yi+1], Y[xi+1,yi]
            dy, dx = abs(y0-y1), abs(x0-x1)
            z00, z10, z01, z11 = func(x0,y0)[0],func(x1,y0)[0],func(x0,y1)[0],func(x1,y1)[0]
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


def interpolateSplines(array,rangex=(0,1),rangey=(0,1)):

    # get array shape
    Dshape=np.shape(array)
    Xl=np.linspace(rangex[0],rangex[1],Dshape[0])
    Yl=np.linspace(rangey[0],rangey[1],Dshape[1])

    splines = RectBivariateSpline(Xl,Yl,array)
    return splines

def surfaceArea(funcdata,rangex=(0,1),rangey=(0,1),coordoi=[0,1,0,1],resolution=1000):
    func=funcdata
    if str(type(funcdata))=="<class 'numpy.ndarray'>":
        func = interpolateSplines(funcdata,rangex,rangey)
    if str(type(func))=="<class 'scipy.interpolate.fitpack2.RectBivariateSpline'>":
        x,dx = np.linspace(coordoi[0], coordoi[1], resolution),(coordoi[1]-coordoi[0])/resolution
        y,dy = np.linspace(coordoi[2], coordoi[3], resolution),(coordoi[3]-coordoi[2])/resolution
        X, Y = np.meshgrid(x, y)
        XY=np.vstack([X.ravel(), Y.ravel()]).T
        dfdx=np.square(func(XY[:,0],XY[:,1],dx=1,grid=False))
        dfdy=np.square(func(XY[:,0],XY[:,1],dy=1,grid=False))
        return sum((np.sqrt(dfdx+dfdy+1))*dx*dy)

def surfaceArea2(funcdata,rangex=(0,1),rangey=(0,1),coordoi=[0,1,0,1],resolution=1000):
    ZZ = funcdata.ravel()
    XX = np.linspace(rangex[0], rangex[1], len(ZZ))
    YY = np.linspace(rangey[0], rangey[1], len(ZZ))
    func = interp2d(XX,YY,ZZ,kind='linear')
    x = np.linspace(rangex[0], rangex[1], resolution)
    y = np.linspace(rangey[0], rangey[1], resolution)
    X, Y = np.meshgrid(x, y)
    atotal=0
    for index, _ in np.ndenumerate(X):
        xi,yi = index
        if np.max([xi,yi]) >= resolution-1:
            #print('out of range')
            None
        else:
            #print('in range')
            x0, y0, x1, y1 = X[xi,yi], Y[xi,yi], X[xi,yi+1], Y[xi+1,yi]
            dy, dx = abs(y0-y1), abs(x0-x1)
            z00, z10, z01, z11 = func(x0,y0)[0], func(x1,y0)[0], func(x0,y1)[0], func(x1,y1)[0]
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


def surfaceArea2d(funcdata,resolution=1000):
    func = interp2d(funcdata[:,1],funcdata[:,2],funcdata[:,3],kind='cubic')
    x,dx = np.linspace(min(funcdata[:,1]),max(funcdata[:,1]), resolution),(max(funcdata[:,1])-min(funcdata[:,1]))/resolution
    y,dy = np.linspace(min(funcdata[:,2]),max(funcdata[:,2]), resolution),(max(funcdata[:,2])-min(funcdata[:,2]))/resolution
    # X, Y = np.meshgrid(x, y)
    # XY=np.vstack([X.ravel(), Y.ravel()]).T
    dfdx=np.square(func(x,y,dx=1)[:,2])
    dfdy=np.square(func(x,y,dy=1)[:,2])
    return sum((np.sqrt(dfdx+dfdy+1))*dx*dy)
    
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


ExData={
'baseline':[[1.7990595561479504E-10,1.7990382115961336E-10],[1.7994457964012001E-10,1.7995003532155947E-10],[1.7987028977007651E-10,1.7987028977007651E-10]],
'disp0000':[[1.7990595561479504E-10,1.7983960336554479E-10],[1.7986633042319215E-10,1.7978040133288903E-10],[1.7994939626791074E-10,1.7977506907697439E-10]],
'disp0250':[[1.7982520116627783E-10,1.7992696147950666E-10],[1.7982317907364207E-10,1.7991267307564131E-10],[1.7986762495834726E-10,1.7992552485838057E-10]],
'disp0500':[[1.7989150149950011E-10,1.8020102449183596E-10],[1.7988236807730763E-10,1.8022698090636451E-10],[1.7988502475841392E-10,1.8023060833055646E-10]],
'disp0750':[[1.7995661609463513E-10,1.8082382038834955E-10],[1.7994491126291240E-10,1.8093697690769745E-10],[1.7992963202265918E-10,1.8096940793068973E-10]],
'disp1000':[[1.7996187904032001E-10,1.8149209999999990E-10],[1.7985769693435520E-10,1.8155696414528489E-10],[1.7982097774075308E-10,1.8158589646784407E-10]],
'disp1250':[[1.7999228530489837E-10,1.8212680806397874E-10],[1.7993860189936697E-10,1.8220541002998993E-10],[1.7995450579806736E-10,1.8224538813728752E-10]],
'disp1500':[[1.7996765264911702E-10,1.8237249793402200E-10],[1.7993723765411535E-10,1.8242050713095628E-10],[1.7989168120626451E-10,1.8245334655114967E-10]],
'disp1750':[[1.7994278560479838E-10,1.8315608213928697E-10],[1.7996423238920360E-10,1.8317563115628122E-10],[1.7995721446184608E-10,1.8313084078640450E-10]],
'disp2000':[[1.7976442585804729E-10,1.8374496757747421E-10],[1.7975736634455182E-10,1.8366111932689097E-10],[1.7975862845718098E-10,1.8370550683105627E-10]],
'disp2250':[[1.7995709053648787E-10,1.8424358840386548E-10],[1.7989214501832720E-10,1.8427068393868700E-10],[1.7986785534821724E-10,1.8421461416194598E-10]],
'disp2500':[[1.7991031246251259E-10,1.8481142612462524E-10],[1.7987678797067646E-10,1.8481753922025981E-10],[1.7983362579140281E-10,1.8481595098300564E-10]],
'disp2750':[[1.7988721586137955E-10,1.8575111739420199E-10],[1.7988415924691768E-10,1.8580503302232586E-10],[1.7984923508830389E-10,1.8575220309896695E-10]],
'disp3000':[[1.7995552769076969E-10,1.8664600893035660E-10],[1.7995420116627788E-10,1.8660680559813389E-10],[1.7994752475841388E-10,1.8661573498833732E-10]]
}

dcoc1 = []
dcoc2 = []
dcoc3 = []
for key, trials in ExData.items():
    t1 , t2, t3 = trials
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
    
                
    undeform.update({F_name[file]:SEC_undeform_plane})
    deform.update({F_name[file]:SEC_deform_plane})
    print(np.size(SEC_deform_plane))
    
    undeformedArea=surfaceIntegrationPlane(SEC_undeform_planeX,
                                           SEC_undeform_planeY,
                                           SEC_undeform_plane)
    
    deformedArea=surfaceIntegrationPlane(SEC_deform_planeX,
                                         SEC_deform_planeY,
                                         SEC_deform_plane)
    
    
    # margin = round(region - .5,1)
    # rangex=(min(SEC_undeform[:,1]),max(SEC_undeform[:,1]))
    # rangey=(min(SEC_undeform[:,2]),max(SEC_undeform[:,2]))
    # cordoi=[rangex[0]+margin,rangex[1]-margin,rangey[0]+margin,rangey[1]-margin ]
    #cordoi=[-.5,.5,-.5,.5]
    #undeformedArea=surfaceArea2(SEC_undeform_plane,rangex,rangey,cordoi,resolution=300)
    
    #undeformedArea=surfaceIntegration(SEC_undeform_sorted,21)
    
    # rangex=(min(SEC_deform[:,1]),max(SEC_deform[:,1]))
    # rangey=(min(SEC_deform[:,2]),max(SEC_deform[:,2]))
    # cordoi=[rangex[0]+margin,rangex[1]-margin*1.0,rangey[0]+margin,rangey[1]-margin ]
    #deformedArea=surfaceArea2(SEC_deform_plane,rangex,rangey,cordoi,resolution=300)
    #deformedArea=surfaceIntegration(SEC_deform_sorted,21)
                
    # undeformedArea=surfaceArea2d(SEC_undeform)
    # deformedArea=surfaceArea2d(SEC_deform)
    
    # plot_surf(SEC_undeform_plane)
    # plot_surf(SEC_deform_plane)
    
    abadcoc.update({F_name[file]:((deformedArea**2)/(undeformedArea**2)-1)})
    
    print(F_name[file])
    print('Predicted change in capacitance from abaqus:')
    print(((deformedArea**2)/(undeformedArea**2)-1)/10)
    print('\n')
    
    # print('Measured change in capacitance:')
    # print(dcoc[F_name[file]])
    
    
xspan=25.4*np.array([0,.025,.05,.075,.1,.125,.15,.175,.2,.225,.25,.275,.3])   
plt.figure()
measure_dcoc1 = dcoc1[1:]
measure_dcoc2 = dcoc2[1:]
measure_dcoc3 = dcoc3[1:]
abaqus_dcoc = np.append([0],[abadcoc[file] for file in F_name])
plt.plot(xspan,measure_dcoc1)
plt.plot(xspan,measure_dcoc2)
plt.plot(xspan,abaqus_dcoc)


m,b=np.polyfit(xspan,abaqus_dcoc,deg=1)
yspan=m*xspan + b 
onedegmse = np.average(np.square(yspan-abaqus_dcoc))
print('Error 1deg: '+str( onedegmse ))
print('\n')

plt.figure()
plt.plot(xspan,abaqus_dcoc)
plt.plot(xspan,yspan)


# m1, m0, b = np.polyfit(xspan,abaqus_dcoc,deg=2)
# yspan=m1*(xspan**2)+m0*xspan+b

m, b = np.polyfit(xspan,abaqus_dcoc,deg=1)
yspan= m*xspan+b
twodegmse = np.average(np.square(yspan-abaqus_dcoc))
print('Error 2deg: '+str( twodegmse ))
print('\n')

QoFit1v2=(twodegmse-onedegmse)/onedegmse
QoFit2v1=(onedegmse-twodegmse)/twodegmse
print('Percent change error 1deg to 2deg: '+str(QoFit1v2))
print('\n')
print('Percent change error 2deg to 1deg: '+str(QoFit2v1))
print('\n')

plt.figure()
plt.plot(xspan,abaqus_dcoc)
plt.plot(xspan,yspan)

plt.figure()
plt.plot(xspan,measure_dcoc1)
plt.plot(xspan,measure_dcoc2)
plt.plot(xspan,measure_dcoc3)
plt.plot(xspan,yspan)


# exp_mean_dcoc=[np.average([measure_dcoc1[i],measure_dcoc2[i],measure_dcoc3[i]]) for i in range(len(xspan))]
# expfitm2,expfitm1,b=np.polyfit(xspan,exp_mean_dcoc,deg=2)
# exp_yspan=expfitm2*(xspan**2)+expfitm1*xspan+b

exp_mean_dcoc=[np.average([measure_dcoc1[i],measure_dcoc2[i],measure_dcoc3[i]]) for i in range(len(xspan))]
expfitm,b=np.polyfit(xspan,exp_mean_dcoc,deg=1)
exp_yspan=expfitm*xspan+b

res=[]
for i in range(len(xspan)):
    res.append(np.std([np.abs(measure_dcoc1[i]-exp_yspan[i]), np.abs(measure_dcoc2[i]-exp_yspan[i]), np.abs(measure_dcoc3[i]-exp_yspan[i])]))
    
res_std = np.average(res)
res_ci_98 = 5.841*res_std/(12**0.5) 


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

mse_min = float('inf')
for i in np.linspace(-50,50,10000):
    mse = np.sum(np.square(np.subtract(yspan,i*exp_yspan)))
    if mse < mse_min:
        factor , mse_min = i, mse
        
plt.figure(figsize=(8,5))
#plt.plot(xspan,factor*exp_yspan)
plt.scatter(xspan,factor*np.array(measure_dcoc1),color='tab:orange')
plt.scatter(xspan,factor*np.array(measure_dcoc2),color='tab:orange')
plt.scatter(xspan,factor*np.array(measure_dcoc3),color='tab:orange')
plt.plot(xspan,factor*(exp_yspan+res_ci_98), 'g--',label=r'expermental mean $\pm$ C.I. 99%')
plt.plot(xspan,factor*(exp_yspan-res_ci_98), 'g--')
plt.plot(xspan,yspan,color='steelblue',label='abaqus model')
plt.xlabel('deflection by boundary (mm)')
plt.ylabel('capacitance '+r'$(\Delta\%)$')
plt.title('gauge factor of '+str(round(factor,3)))
plt.legend()
plt.tight_layout()
plt.savefig('JigEx_Model.png',dpi=300)

print(factor)


plt.figure()
plt.scatter(xspan,measure_dcoc1,c='tab:orange',label="experimental measure")
plt.scatter(xspan,measure_dcoc2,c='tab:orange')
plt.scatter(xspan,measure_dcoc3,c='tab:orange')
plt.plot(xspan,yspan,color='steelblue',label='abaqus model')
plt.xlabel('deflection by boundary (mm)')
plt.ylabel('capacitance '+r'$(\Delta\%)$')
#plt.title('gauge factor of '+str(round(factor,3)))
plt.legend()
plt.tight_layout()





























