# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 11:36:41 2022

@author: AVEREEN
"""

import IPython as IP
IP.get_ipython().magic('reset -sf')

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp2d

import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import linear_model
from sklearn import pipeline
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm

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

def surfaceArea2d(funcdata,resolution=100):
    resolution=20
    func = interp2d(funcdata[:,1],funcdata[:,2],funcdata[:,3],kind='cubic')
    X,dx = np.linspace(min(funcdata[:,1]),max(funcdata[:,1]), resolution),(max(funcdata[:,1])-min(funcdata[:,1]))/resolution
    Y,dy = np.linspace(min(funcdata[:,2]),max(funcdata[:,2]), resolution),(max(funcdata[:,2])-min(funcdata[:,2]))/resolution
    # X, Y = np.meshgrid(x, y)
    # X, Y = X.ravel(), Y.ravel()
    dfdx = np.square(func(X,Y,dx=1).ravel())
    dfdy = np.square(func(X,Y,dy=1).ravel())
    return (np.sqrt(dfdx+dfdy+1))*dx*dy


resolution=100
def surfaceIntegration(Z,resolution=20):
    func = interp2d(Z[:,1],Z[:,2],Z[:,3],kind='linear')
    x,dx = np.linspace(min(Z[:,1]),max(Z[:,1]), resolution),(max(Z[:,1])-min(Z[:,1]))/resolution
    y,dy = np.linspace(min(Z[:,2]),max(Z[:,2]), resolution),(max(Z[:,2])-min(Z[:,2]))/resolution
    X, Y = np.meshgrid(x, y)
    atotal=0
    for index, _ in np.ndenumerate(X):
        xi,yi = index
        if np.max([xi,yi]) >= resolution-1:
            #print('out of range')#
            None
        else:
            #print('in range')
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




pp_p=0.43
sec_p=0.3
guage=(1-pp_p)/(1-sec_p)


res = 20
node = np.array(list(range(res**2))).T
x, y = np.meshgrid(np.linspace(0,1,num=res), np.linspace(0,1,num=res))
X, Y = x.ravel(), y.ravel()
#points=np.array([(y,x) for x,y in zip(X,Y)]).reshape(x.shape)
zz = np.ones_like(X).ravel().T
funcdata=np.array([node,X,Y,zz]).T
Z=np.array([node,X,Y,zz]).T
func = interp2d(Z[:,1],Z[:,2],Z[:,3],kind='cubic')
area=surfaceIntegration(funcdata,resolution=20)




    
    
    
    
    