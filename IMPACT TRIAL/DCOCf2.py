# -*- coding: utf-8 -*-#
"""
Created on Thu Aug 26 11:10:12 2021

@author: AVEREEN
"""

#%% Imports and Configurations

import IPython as IP
IP.get_ipython().magic('reset -sf')

from os.path import exists
import numpy as np
import scipy as sp
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import matplotlib.colors as clr
import sklearn as sk
from sklearn import linear_model
from sklearn import pipeline



from matplotlib.patches import Patch
from matplotlib.lines import Line2D


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

#%% Functions

def consecutive(ar, step=1):
    return np.split(ar, np.where(np.diff(ar) != step)[0]+1)

def strip_plot(tt,du,fi,la): # numpy
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

def grab_cap_region(tt,du,event,buffer=1,tail=10):
    import numpy as np
    # time points
    p1=event-tail; p2=event-buffer;
    a1=event+buffer; a2=event+tail;
    
    # pre impact region
    cc1,tt1=strip_plot(tt,du,p1,p2)
    ccm1,ccb1=np.polyfit(tt1,cc1,1)
    #cclinfac1=ccm1*tt1
    cc1nd=cc1#-cclinfac1 #remove drift
    
    # after impact region
    cc2,tt2=strip_plot(tt,du,a1,a2)
    ccm2,ccb2=np.polyfit(tt2,cc2,1)
    #cclinfac2=ccm2*tt2
    cc2nd=cc2#-cclinfac2 #remove drift
    tt2=tt2+(a1-p1) #time displacement
    
     
    # total region
    cc,tt=strip_plot(tt,du,p1,a2)
    ccm,ccb=np.polyfit(tt,cc,1)
    cclinfac=np.average([ccm1,ccm2])
    ccnd=cc-cclinfac*tt #remove drift
    cc2nd=cc2-cclinfac*tt2
    cc1nd=cc1-cclinfac*tt1 
    
    # package
    region=np.stack([tt,ccnd],axis=1)
    prior=np.stack([tt1,cc1nd],axis=1)
    posterier=np.stack([tt2,cc2nd],axis=1)
    
    return region, prior, posterier

def trim_1d(array,time):
    buff=4
    dt=1/14997.7
    flags = consecutive(np.flatnonzero(array))
    
    splits = [[flag[0],flag[-1]] for flag in flags]
    
    if len(splits) > 2: 
        peaks =[]
        for split in splits:
            du,tt= strip_plot(time,array,time[split[0]],time[split[1]]+.01)
            peaks = np.append(peaks,np.max(du))
        peaks[peaks<.5*np.max(peaks)] = 0
        
        for p in range(0,len(peaks)):
            if peaks[p] == 0 and len(splits)>2:
                np.delete(splits,p)
            else:
                pass
    else:
        pass
    
    if len(splits) > 1:
        start=splits[-1]
        end=splits[0]
        loadStart, timeStart = strip_plot(time,array,time[start[0]-buff],time[start[1]])
        loadEnd, timeEnd = strip_plot(time,array,time[end[0]],time[end[1]+buff])
        load=np.append(loadStart,loadEnd)
        #t_total= (time[start[1]]-time[start[0]-buff])+(time[end[1]+buff]-time[end[0]])
        t_total = dt*len(load)
        tt = np.linspace(0,t_total,num=len(load))+time[start[1]]
    elif len(splits) == 1:
        start, stop = splits[0]
        load, tt = strip_plot(time,array,time[start-buff],time[stop+buff])
        t_total = dt*len(load)
        tt = np.linspace(0,t_total,num=len(load))+time[start]
        #tt += time[start]
    else:
        load = array
        tt = time
        print('More than two nonzero regions')
    return load , tt

def scale_correction(pload,pscale=-10.25,pbias=-0.65,nscale=-15.5621,nbias=-0.615633):
    Tload=nscale*((pbias-nbias)+pload/pscale)
    return Tload

def grab_load_region(tt,du,event,buffer=0.2,tail=2,MASSIMP=7.4752023):
    import numpy as np
    # Time points
    base1 = event-buffer-tail; base2 = event-buffer;
    imp1 = event-buffer; imp2 = event+buffer;
    
    # Pre impact region
    baseline, tt_base = strip_plot(tt,du,base1,base2)
    base_avg = np.mean(baseline)

    # Impact region
    Impact, tt_imp = strip_plot(tt,du,imp1,imp2)
    Impact -= base_avg+np.ptp(baseline)*5
    Impact[Impact < 0] = 0
    Impact, tt_imp = trim_1d(Impact,tt_imp)
    Impact += base_avg+np.ptp(baseline)*5

    #tt_imp += imp1 #time displacement
    
    # Package
    region=np.stack([tt_imp,Impact],axis=1)
    prior=np.stack([tt_base,baseline],axis=1)
    
    # Integrate region
    dp = np.trapz(Impact,tt_imp)
    dv = dp/MASSIMP
    #dvr, dvre = e_parse(region)
    
    
    return region, prior, dv/2

def e_parse(region,MASSIMP=7.4752023):                
    tt=region[:,0]
    du=region[:,1]
    ttmax=tt[np.argmax[du]]
    ramp, ttr = strip_plot(tt,du,tt[0],ttmax)
    release, ttre = strip_plot(tt,du,ttmax,tt[-1])
    dpr = np.trapz(ramp,ttr)
    dvr = dpr/MASSIMP
    dpre = np.trapz(release,ttre)
    dvre = dpre/MASSIMP
    return dvr, dvre

def explicit_plastic(speed,region,m=7.4752023,g=9.81,dx=(36.33/1000),tscale=0.000001):
    At, Vt, Dt, tt =  time_integrator(region,g=9.81,m=m,di=0,vi=speed)
    return (m/2)*(Vt[0]**2-Vt[-1]**2)+m*g*Dt[-1], ( At, Vt, Dt, tt)

def time_integrator(region,g=9.81,m=1,di=0,vi=0):
    tt=region[:,0];tt-=tt[0];
    du=region[:,1]/m;du-=du[0];
    dum=(du[-1]-du[0])/((tt[-1]-tt[0]))
    du = [du[i]-dum*tt[i] for i in range(len(tt))]
    
    
    du_interpolate = interpolate.interp1d(tt, du ,kind='cubic')
    tt = np.linspace(tt[0],tt[-1],num=int(len(tt)*10))
    step = (tt[-1]-tt[0])/(len(tt));
    a = lambda x: du_interpolate(x)
    
    At=[g]
    At=np.concatenate((At,[g-a(t) for t in tt]))
    At=np.ravel(np.concatenate((At,[g])))
    At=list(At)
    tt=np.append([0], tt+step )
    tt=np.append(tt, tt[-1]+step )
    
    #At.reverse()
    Vt=[vi]
    for k in range(1,len(At)-1):
        Vt=np.append(Vt,Vt[k-1]+(1/6)*(At[k-1]+4*At[k]+At[k+1])*2*step)
    #plot_this(ttv, Vt, 'time', 'velocity')
    
    Dt=[di,float('-inf')]
    tune = 0
    while Dt[-1] < 0:
        Dt=[di]
        vcross=np.argmin(At)+tune
        Vb = Vt[vcross]
        Vm = Vt[0]/(Vt[0]-Vb)
        Vtn = [Vm*(Vt[i]-Vb) for i in range(len(Vt))]
        for k in range(1,len(Vtn)-1):
            Dt=np.append(Dt,Dt[k-1]+(1/6)*(Vtn[k-1]+4*Vtn[k]+Vtn[k+1])*2*step)
        tune+=1
    return At, Vtn, Dt , tt
        
def integrateVisual(Region,key):
    yls=['distance','velocity','accelaration']
    Time=Region[3]
    Region=[Region[0],Region[1],Region[2]]
    for R in Region:
        yl=yls.pop(-1)
        tt=np.linspace(Time[0],Time[-1],num=int(len(R)))
        plot_this(tt,R,'time',yl)
        plt.savefig('KINE/'+yl+key+'.png')

def plot_these(tt,du,xl,yl):
    fig=plt.figure(figsize=(6.5,4))
    for i in range(0,len(tt)):
       plt.plot(tt[i],du[i],'b^')
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.tight_layout()
    plt.savefig('cmp.pdf')
    return fig

def plot_this(tt,du,xl,yl):
    fig=plt.figure(figsize=(6.5,4))
    plt.plot(tt,du)
    plt.xlabel(xl)
    plt.ylabel(yl)
    return fig

def dt2en(time,skey,MASSIMP=7.4752023,dx=(36.33/1000),tscale=0.000001):
    ENERGY={}
    SPEED={}
    for key in skey:
        SP=dx/(tscale*float(time[key]))
        EN=0.5*MASSIMP*(SP**2)
        ENERGY.update({key:EN})
        SPEED.update({key:SP})
    return ENERGY,SPEED

def work(vi,dv,MASSIMP=7.4752023):
    e1=0.5*MASSIMP*(vi**2)
    e2=0.5*MASSIMP*((dv-vi)**2)
    work=abs(e1-e2)
    rebound_HeightCm = e2/(MASSIMP*9.81)*100
    return work, e1, e2 , rebound_HeightCm
 
#%% Declarations

C, L, N = [], [], []

lb2N = 4.4482216153

nompr = 5.084321175

Tcal = [-0.615633,-15.5621]

SEC={'1':'25CM_PN1/cap.lvm',#PN1
      '2':'25CM_PN2/cap.lvm',#PN2
      '3':'30CM_PN3/cap.lvm',#PN3
      '4':'30CM_PN4/cap.lvm',#PN4
      '5':'SEC29_20CM_N2/cap.lvm',#N2
      '6':'SEC27_20CM_N4/cap.lvm',#N4
      '7':'SEC25_18CM_N6/cap.lvm',#N6
      '8':'SEC24_15CM_N7/cap.lvm',#N7
      '9':'SEC23_15CM_N8/cap.lvm',#N8 
      '10':'SEC22_12.5CM_N10/cap.lvm',#N10
      '11':'SEC21_12.5CM_N11/cap.lvm',#N11
      '12':'SEC19_10CM_N12/cap.lvm',#N12
      '13':'SEC20_10CM_N13/cap.lvm',#N13
      '14':'SEC18_25CM_N14/cap.lvm',#N14
      '15':'SEC16_25CM_N15/cap.lvm',#N15
      '16':'SEC17_22.5CM_N16/cap.lvm',#N16
      '17':'SEC15_22.5CM_N17/cap.lvm',#N17
      '18':'SEC14_27.5CM_N18/cap.lvm',#N18
      '19':'SEC13_27.5CM_N19/cap.lvm',#N19
      '20':'SEC12_25CM_N20/cap.lvm',#N20
      '21':'SEC11_25CM_N21/cap.lvm',#N21
      '22':'SEC10_30CM_N22/cap.lvm',#N22
      '23':'SEC9_30CM_N23/cap.lvm',#N23
      '24':'SEC8_27.5CM_N24/cap.lvm',#N24
      '25':'SEC7_27.5CM_N25/cap.lvm',#N25
      '26':'SEC6_25CM_N26/cap.lvm',#N26
      '27':'SEC5_25CM_N27/cap.lvm',#N27
      '28':'SEC4_25CM_N28/cap.lvm',#N28
      '29':'SEC3_22.5CM_N29/cap.lvm',#N29
      '30':'SEC2_25CM_N30/cap.lvm',#N30
      '31':'SEC1_25CM_N31/cap.lvm',#N31
      }

LOAD={'1':'25CM_PN1/force.lvm',#PN1
      '2':'25CM_PN2/force.lvm',#PN2
      '3':'30CM_PN3/force.lvm',#PN3
      '4':'30CM_PN4/force.lvm',#PN4
      '5':'SEC29_20CM_N2/force.lvm',#N2
      '6':'SEC27_20CM_N4/force.lvm',#N4
      '7':'SEC25_18CM_N6/force.lvm',#N6
      '8':'SEC24_15CM_N7/force.lvm',#N7
      '9':'SEC23_15CM_N8/force.lvm',#N8 
      '10':'SEC22_12.5CM_N10/force.lvm',#N10
      '11':'SEC21_12.5CM_N11/force.lvm',#N11
      '12':'SEC19_10CM_N12/force.lvm',#N12
      '13':'SEC20_10CM_N13/force.lvm',#N13
      '14':'SEC18_25CM_N14/force.lvm',#N14
      '15':'SEC16_25CM_N15/force.lvm',#N15
      '16':'SEC17_22.5CM_N16/force.lvm',#N16
      '17':'SEC15_22.5CM_N17/force.lvm',#N17
      '18':'SEC14_27.5CM_N18/force.lvm',#N18
      '19':'SEC13_27.5CM_N19/force.lvm',#N19
      '20':'SEC12_25CM_N20/force.lvm',#N20
      '21':'SEC11_25CM_N21/force.lvm',#N21
      '22':'SEC10_30CM_N22/force.lvm',#N22
      '23':'SEC9_30CM_N23/force.lvm',#N23
      '24':'SEC8_27.5CM_N24/force.lvm',#N24
      '25':'SEC7_27.5CM_N25/force.lvm',#N25
      '26':'SEC6_25CM_N26/force.lvm',#N26
      '27':'SEC5_25CM_N27/force.lvm',#N27
      '28':'SEC4_25CM_N28/force.lvm',#N28
      '29':'SEC3_22.5CM_N29/force.lvm',#N29
      '30':'SEC2_25CM_N30/force.lvm',#N30
      '31':'SEC1_25CM_N31/force.lvm',#N31
      }

TIME={'1':16000.00,#PN1
      '2':15616.00,#PN2
      '3':13888.00,#PN3
      '4':16896.00,#PN4
      '5':25600.00,#N2
      '6':27648.00,#N4
      '7':26404.00,#N6
      '8':28928.00,#N7
      '9':28928.00,#N8 
      '10':35328.00,#N10
      '11':36608.00,#N11
      '12':40704.00,#N12
      '13':43264.00,#N13
      '14':27904.00,#N14
      '15':23552.00,#N15
      '16':26368.00,#N16
      '17':29952.00,#N17
      '18':19712.00,#N18
      '19':26368.00,#N19
      '20':21504.00,#N20
      '21':19840.00,#N21
      '22':19968.00,#N22
      '23':19840.00,#N23
      '24':19968.00,#N24
      '25':20160.00,#N25
      '26':20328.00,#N26
      '27':20224.00,#N27
      '28':21248.00,#N28
      '29':23040.00,#N29
      '30':22784.00,#N30
      '31':22784.00,#N31
      }

TIME_STAMP={'1':['13h  30m  20.6146s','13h  34m  19.7896s'],
            '2':['13h  55m  37.2970s','13h  59m  06.6238s'],
            '3':['14h  24m  46.9077s','14h  29m  24.3674s'],
            '4':['14h  56m  17.3155s','15h  01m  09.0697s'],
            '5':['17h  16m  27.7127s','17h  19m  04.7118s'],
            '6':['19h  03m  45.6453s','19h  06m  57.7799s'],
            '7':['15h  46m  06.9845s','15h  49m  04.1034s'],
            '8':['16h  38m  16.6335s','16h  42m  17.7388s'],
            '9':['16h  48m  59.2420s','16h  53m  04.7242s'],
            '10':['18h  04m  52.2791s','18h  08m  07.1460s'],
            '11':['17h  29m  53.6825s','17h  33m  55.4245s'],
            '12':['18h  38m  25.4838s','18h  41m  05.2749s'],
            '13':['18h  47m  05.4590s','18h  51m  12.5506s'],
            '14':['12h  47m  17.4926s','12h  52m  04.3537s'],
            '15':['13h  14m  30.8738s','13h  18m  37.6227s'],
            '16':['14h  06m  37.6456s','14h  10m  11.4003s'],
            '17':['14h  23m  15.0170s','14h  28m  12.8743s'],
            '18':['16h  25m  40.0874s','16h  29m  43.6719s'],
            '19':['14h  54m  24.4794s','14h  58m  21.4165s'],
            '20':['12h  19m  13.4063s','12h  25m  14.7435s'],
            '21':['12h  32m  13.1542s','12h  36m  33.0255s'],
            '22':['13h  56m  58.7309s','14h  01m  30.4241s'],
            '23':['13h  26m  32.8137s','13h  30m  27.3105s'],
            '24':['14h  19m  28.8033s','14h  24m  47.8728s'],
            '25':['14h  46m  08.2668s','14h  50m  15.9139s'],
            '26':['16h  52m  59.0341s','16h  57m  12.0566s'],
            '27':['17h  33m  11.1868s','17h  38m  10.5712s'],
            '28':['14h  51m  30.6608s','14h  56m  08.1724s'],
            '29':['15h  22m  20.1667s','15h  26m  44.0369s'],
            '30':['16h  12m  40.2278s','16h  17m  36.6923s'],
            '31':['17h  29m  53.5535s','17h  34m  52.5455s']}

B_Sc={'1':[-0.65,-10.25],#PN1
      '2':[-0.65,-10.25],#PN2
      '3':[-0.65,-10.25],#PN3
      '4':[-0.65,-10.25],#PN4
      '5':[-0.66,-14.11],#N2
      '6':[-0.69,-11.25],#N4
      '7':[-0.68729,-12.4436],#N6
      '8':[-0.68729,-12.4436],#N7
      '9':[-0.68729,-12.4436],#N8 
      '10':[-0.69,-12.4436],#N10
      '11':[-0.69,-12.4436],#N11
      '12':[-0.69,-12.4436],#N12
      '13':[-0.69,-12.4436],#N13
      '14':[-0.715,-11.9],#N14
      '15':[-0.715,-11.9],#N15
      '16':[-0.715,-11.9],#N16
      '17':[-0.715,-11.9],#N17
      '18':[-0.715,-11.9],#N18
      '19':[-0.715,-11.9],#N19
      '20':[-0.715,-11.9],#N20
      '21':[-0.715,-11.9],#N21
      '22':[-0.715,-11.9],#N22
      '23':[-0.715,-11.9],#N23
      '24':[-0.715,-11.9],#N24
      '25':[-0.725,-11.9],#N25
      '26':[-0.725,-11.9],#N26
      '27':[-0.725,-11.9],#N27
      '28':[-0.71,-11],#N28
      '29':[-0.71,-11],#N29
      '30':[-0.71,-11],#N30
      '31':[-0.71,-11],#N31
      }

skey=[str(key) for key in np.arange(0,len(SEC),step=1)+1]

ENERGY,SPEED=dt2en(TIME,skey)

#%% Time  

TIME={}
for key in skey:
    disallow='hms'
    times=TIME_STAMP[key]
    t1=times[0]; t2=times[1];
    for char in disallow:
        t1=t1.replace(char,'')
        t2=t2.replace(char,'')
    t1=np.asarray(t1.split(),dtype=float)
    t2=np.asarray(t2.split(),dtype=float)
    t1=t1[0]*3600+t1[1]*60+t1[2]
    t2=t2[0]*3600+t2[1]*60+t2[2]
    TIME.update({key:t2-t1})

#%% Load Cap Data
    
CAP_files={}
EVENT_time={}
for key in skey:
    D=np.genfromtxt(SEC[key],skip_header=22,delimiter='\t')
    du=np.asarray(D[:,1],dtype=float)
    ttu=np.linspace(0,TIME[key],num=len(du))
    du, ttu= strip_plot(ttu, du, 35, ttu[-1])
    EVENT_time.update({key:ttu[np.argmax(np.abs(np.diff(du)))+1]})    
    CAP_files.update({key:np.stack([ttu,du],axis=1)})

#%% Plot Cap Data
CAP_REGIION={}
DCOC={}
for key in skey:
    D=CAP_files[key]
    event=EVENT_time[key]
    R,P,PO = grab_cap_region(D[:,0],D[:,1],event)
    CAP_REGIION.update({key:(R-np.mean(P[:,1]))/np.mean(P[:,1])})
    plot_this(R[:,0], R[:,1], 'time (s)', yl)
    plt.plot(P[:,0], P[:,1])
    plt.plot(PO[:,0], PO[:,1])
    plt.savefig(SEC[key].replace('cap.lvm','cap.png'),dpi=300)
    plt.close('all')
    DCOC.update({key:(np.mean(PO[:,1])-np.mean(P[:,1]))/np.mean(P[:,1])})
    path=SEC[key].replace('cap.lvm','cap_trim.txt')
    np.savetxt(path,R)
del(R,P,PO,D,event,key,path)    


#%% Load Force Data
flush=False
LOAD_files={}
EVENT_time={}
clipS=0#45
clipE=0#30
for key in skey:
    if exists(LOAD[key].replace('force.lvm','force_trim.txt')) and not flush:
        D=np.genfromtxt(LOAD[key].replace('force.lvm','force_trim.txt'),delimiter=' ',dtype=np.float32)
        du=np.asarray(D[:,1],dtype=np.float32)
        ttu=np.asarray(D[:,0],dtype=np.float32)
        event=ttu[np.argmax(np.abs(np.diff(du)))+1]
        del(D)
    elif exists(LOAD[key]):
        bs=B_Sc[key]
        D=np.genfromtxt(LOAD[key],skip_header=22,delimiter='\t',dtype=np.float32)
        du=np.asarray(D[:,1],dtype=np.float32)
        del(D)
        ttu=np.linspace(0,TIME[key],num=len(du))
        du, ttu= strip_plot(ttu, du, clipS, ttu[-1]-clipE)
        event=ttu[np.argmax(np.abs(du))]
        du, ttu= strip_plot(ttu, du, event-6, event+2)
        plot_this(ttu, du, xl, yl)
        plt.savefig(LOAD[key].replace('/force.lvm',' force.png'),dpi=300)
        plt.close('all')
        event=ttu[np.argmax(np.abs(np.diff(du)))+1]
        du=np.asarray(du,dtype=np.float32)
        du=scale_correction(du,pscale=bs[1],pbias=bs[0],nscale=Tcal[1],nbias=Tcal[0])*lb2N
        path=LOAD[key].replace('force.lvm','force_trim.txt')
        np.savetxt(path,np.stack([ttu,du],axis=1))
    else:
        print(LOAD[key]+' does not exist')
    EVENT_time.update({key:event}) 
    LOAD_files.update({key:np.stack([ttu,du],axis=1)})
del(du,ttu,event,key)
    

#%% Plot Force Data

skey=[str(key) for key in np.arange(0,len(SEC)-6,step=1)+6]
xl="time " + r"$\left( \mathrm{s} \right)$"
yl="force " + r"$\left( \mathrm{N} \right)$"

IMPACT={}
LOAD_work={}
EXIT_en={}
START_en={}
BOUNCE={}
DELV={}
KINE={}
for key in skey:
    D=LOAD_files[key]
    event=EVENT_time[key]
    R,P,DelV = grab_load_region(D[:,0],D[:,1],event)
    IMPACT.update({key:R})
    DELV.update({key:DelV})
    # plot_this(R[:,0], R[:,1], 'time (s)', 'Force (N)') 
    # plt.savefig(LOAD[key].replace('force.lvm','force.png'),dpi=300)
    # plt.close('all')
    Work , kine = explicit_plastic(SPEED[key], R)
    integrateVisual(kine,key)
    plt.close('all')
    WORK, EN1, EN2 , bounce = work(SPEED[key],DelV)
    START_en.update({key:EN1})
    EXIT_en.update({key:EN2})
    LOAD_work.update({key:Work})
    BOUNCE.update({key:bounce})
    KINE.update({key:kine})
    
out='Study result:'
for sample in skey:
    output=str('\nFor Sample '+str(sample)+': '
               +'\n  Impact energy: '+str(START_en[sample])
               +'\n  Departure energy: '+str(EXIT_en[sample])
               +'\n  Work done by plate: '+str(LOAD_work[sample])
               +'\n  Rebound height cm: '+str(BOUNCE[sample])
               +'\n  Normal Delta C: '+str(DCOC[sample])+'\n')
    out+=output
f = open('output.txt', "w")
f.write(out)
f.close()

# =============================================================================
# skey=[str(key) for key in np.arange(0,len(SEC)-6,step=1)+6]
# =============================================================================

plot_ken = False
if plot_ken:
    fig=plt.figure(figsize=(6,4))
    plt.title('kinetic impact energy')
    C=np.asarray([DCOC[key] for key in skey ])
    start_en=np.asarray([START_en[key] for key in skey ])
    plt.plot(start_en,C,'o',c='tab:green',label="cap")
    plt.ylabel('change in capacitance')
    plt.xlabel('energy (j)')
    plt.ylim(0,np.max(C)*1.15)
    plt.ticklabel_format(axis='y',style='plain', scilimits=(0,0)) #,style='sci'
    plt.tight_layout()
    plt.savefig('impact energy test_veiw.png',dpi=400)

plot_work = False
if plot_work:
    fig=plt.figure(figsize=(6,4))
    plt.title('work done on the plate')
    C=np.asarray([DCOC[key] for key in skey ])
    load_en=np.asarray([LOAD_work[key] for key in skey ])
    plt.plot(1+(load_en-nompr)/nompr,C,'o',c='tab:green',label="cap")
    plt.ylabel('change in capacitance')
    plt.xlabel('energy (j)')
    plt.ylim(0,np.max(C)*1.15)
    plt.ticklabel_format(axis='y',style='plain', scilimits=(0,0)) #,style='sci'
    plt.tight_layout()
    plt.savefig('Work done test_veiw.png',dpi=400)

plot_dual = False
if plot_dual:
    C=np.asarray([DCOC[key]/2 for key in skey ])
    start_en=np.asarray([START_en[key] for key in skey ])
    load_en=np.asarray([LOAD_work[key] for key in skey ])
    up=5.084321175
    med=2.881115333
    x0, x1 = np.meshgrid(np.linspace(0,start_en[-1]*2,800),np.linspace(0,1,2))
    z = x0*0
    for ix,iy in np.ndindex(x0.shape):
        if x0[ix,iy]>up:
            z[ix,iy]=10
        elif x0[ix,iy]<=up and x0[ix,iy]>=med:
            z[ix,iy]=5
        else:
            z[ix,iy]=0
    legend_elements = [Patch(facecolor='bisque', edgecolor='k',label='safe',alpha=.5),
                       Patch(facecolor='gold', edgecolor='k',label='marginal',alpha=.5),
                       Patch(facecolor='tab:orange', edgecolor='k',label='unsafe',alpha=.5)]
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(9,4))
    fig.text(0.28, 0.94, 'impact energy', ha='center', va='center')
    fig.text(0.72, 0.94, 'energy absorbed', ha='center', va='center')
    fig.text(0.08, 0.07, 'a)', ha='center', va='center')
    fig.text(0.52, 0.07, 'b)', ha='center', va='center')
    ax1.plot(start_en,C,'o',c='k',label="cap")
    ax1.contourf(x0,x1,z,levels=3,colors=['bisque','gold','tab:orange','tab:orange'],alpha=.5)
    #ax2.plot(1+(load_en-nompr)/nompr,C,'o',c='tab:green',label="cap")
    ax2.plot(load_en,C,'o',c='k',label="cap")
    ax2.contourf(x0,x1,z,levels=3,colors=['bisque','gold','tab:orange','tab:orange'],alpha=.5)
    fig.text(0.28, 0.04, 'energy before impact (J)', ha='center', va='center')
    fig.text(0.72, 0.04, 'energy absorbed by the plate (J)', ha='center', va='center')
    fig.text(0.03, 0.55, 'capacitance '+r'$(\Delta\%)$', ha='center', va='center', rotation='vertical')
    ax1.set_ylim(0,np.max(C)*1.15)
    ax1.set_xlim(.9,np.max(start_en)*1.15)
    ax2.set_ylim(0,np.max(C)*1.15)
    ax2.set_xlim(.9,np.max(load_en)*1.15)
    ax1.ticklabel_format(axis='y',style='plain', scilimits=(0,0)) #,style='sci'
    ax2.ticklabel_format(axis='y',style='plain', scilimits=(0,0)) #,style='sci'
    ax1.legend(handles=legend_elements, loc=0,framealpha=1)
    plt.tight_layout(rect=(.05,.05,.95,.95))
    plt.savefig('combinedfig.png',dpi=400)


plot_dual = True
if plot_dual:
    C=np.asarray([DCOC[key]/2 for key in skey ])
    start_en=np.asarray([START_en[key] for key in skey ])
    load_en=np.asarray([LOAD_work[key] for key in skey ])
    up=5.084321175
    med=2.881115333
    x0, x1 = np.meshgrid(np.linspace(0,start_en[-1]*2,800),np.linspace(0,1,2))
    z = x0*0
    for ix,iy in np.ndindex(x0.shape):
        if x0[ix,iy]>up:
            z[ix,iy]=10
        elif x0[ix,iy]<=up and x0[ix,iy]>=med:
            z[ix,iy]=5
        else:
            z[ix,iy]=0
    legend_elements = [Patch(facecolor='bisque', edgecolor='k',label='safe',alpha=.5),
                       Patch(facecolor='gold', edgecolor='k',label='marginal',alpha=.5),
                       Patch(facecolor='tab:orange', edgecolor='k',label='unsafe',alpha=.5)]
    fig, ax1 = plt.subplots(nrows=1, ncols=1,figsize=(8,4))
    # fig.text(0.3, 0.94, 'energy absorbed', ha='center', va='center')
    # ax1.plot(start_en,C,'o',c='k',label="cap")
    # ax1.contourf(x0,x1,z,levels=3,colors=['bisque','gold','tab:orange','tab:orange'],alpha=.5)
    #ax2.plot(1+(load_en-nompr)/nompr,C,'o',c='tab:green',label="cap")
    ax1.plot(load_en,C,'o',c='k',label="cap")
    ax1.contourf(x0,x1,z,levels=3,colors=['bisque','gold','tab:orange','tab:orange'],alpha=.5)
    fig.text(0.5, 0.05, 'energy absorbed by the plate (J)', ha='center', va='center')
    fig.text(0.03, 0.55, 'capacitance '+r'$(\Delta\%)$', ha='center', va='center', rotation='vertical')
    # ax1.set_ylim(0,np.max(C)*1.15)
    # ax1.set_xlim(.9,np.max(start_en)*1.15)
    ax1.set_ylim(0,np.max(C)*1.15)
    ax1.set_xlim(.9,np.max(load_en)*1.15)
    ax1.ticklabel_format(axis='y', scilimits=(0,0)) #,style=sci
    ax1.legend(handles=legend_elements, loc=2,framealpha=1)
    plt.tight_layout(rect=(.05,.05,.95,.95))
    plt.savefig('uncombinedfig.png',dpi=400)


plot_picsamp = False
if plot_picsamp:
    plt.rcParams.update({'font.size': 12})
    plt.close('all')
    pic_samples=['13','19','18']
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=len(pic_samples),figsize=(5.82677*2,3))
    tlo,caplo=CAP_REGIION[pic_samples[0]][:,0],CAP_REGIION[pic_samples[0]][:,1]*100
    tmed,capmed=CAP_REGIION[pic_samples[1]][:,0],CAP_REGIION[pic_samples[1]][:,1]*100
    thigh,caphigh=CAP_REGIION[pic_samples[2]][:,0],CAP_REGIION[pic_samples[2]][:,1]*100
    cmax=np.max([np.max(caplo),np.max(capmed),np.max(caphigh)])*.5
    cmin=np.min([np.min(caplo),np.min(capmed),np.min(caphigh)])*1
    ax1.plot(tlo,caplo,'-',c='k',label="caplo")
    ax2.plot(tmed,capmed,'-',c='k',label="capmed")
    ax3.plot(thigh,caphigh,'-',c='k',label="caphigh")
    # exit_en=np.asarray([LOAD_work[key] for key in skey ])
    # ax1.plot(start_en,C,'o',c='k',label="cap")
    # ax1.contourf(x0,x1,z,levels=3,colors=['bisque','gold','tab:orange','tab:orange'],alpha=.5)
    #ax2.plot(1+(load_en-nompr)/nompr,C,'o',c='tab:green',label="cap")
    # ax2.plot(load_en,C,'o',c='k',label="cap")
    # ax2.contourf(x0,x1,z,levels=3,colors=['bisque','gold','tab:orange','tab:orange'],alpha=.5)
    # fig.text(0.28, 0.04, 'energy (j)', ha='center', va='center')
    # fig.text(0.72, 0.04, 'energy (j)', ha='center', va='center')
    fig.text(0.03, 0.55, 'capacitance '+r'$(\Delta\%)$', ha='center', va='center', rotation='vertical')
    ax1.set_ylim(cmin-abs(cmin)*.15,cmax)
    #ax1.set_xlim(.9,np.max(start_en)*1.15)
    ax2.set_ylim(cmin-abs(cmin)*.15,cmax)
    #ax2.set_xlim(.9,np.max(load_en)*1.15)
    ax3.set_ylim(cmin-abs(cmin)*.15,cmax)
    #ax3.set_xlim(.9,np.max(load_en)*1.15)
    ax1.ticklabel_format(axis='y', scilimits=(0,0)) #,style='sci
    ax1.set_xticks([])
    ax2.ticklabel_format(axis='y',style='plain', scilimits=(0,0)) #,style='sci'
    ax2.set_xticks([])
    ax3.ticklabel_format(axis='y',style='plain', scilimits=(0,0)) #,style='sci'
    ax3.set_xticks([])
    fig.text(0.357-.153, 0.04, 'time', ha='center', va='center')
    fig.text(0.674-.153, 0.04, 'time', ha='center', va='center')
    fig.text(0.99-.153, 0.04, 'time', ha='center', va='center')
    fig.text(0.357-.3, 0.04, 'a)', ha='center', va='center')
    fig.text(0.674-.3, 0.04, 'b)', ha='center', va='center')
    fig.text(0.99-.3, 0.04, 'c)', ha='center', va='center')
    #ax1.legend(handles=legend_elements, loc=0)
    plt.tight_layout(rect=(.04,.01,.99,.99))
    plt.savefig('cappicsamp.png',dpi=300)


if plot_picsamp:
    plt.rcParams.update({'font.size': 12})
    plt.close('all')
    pic_samples=['13','19','18']
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=len(pic_samples), ncols=1,figsize=(5.82677*2/3,6))
    tlo,caplo=CAP_REGIION[pic_samples[0]][:,0],CAP_REGIION[pic_samples[0]][:,1]*100
    tmed,capmed=CAP_REGIION[pic_samples[1]][:,0],CAP_REGIION[pic_samples[1]][:,1]*100
    thigh,caphigh=CAP_REGIION[pic_samples[2]][:,0],CAP_REGIION[pic_samples[2]][:,1]*100
    cmax=np.max([np.max(caplo),np.max(capmed),np.max(caphigh)])*.5
    cmin=np.min([np.min(caplo),np.min(capmed),np.min(caphigh)])*1
    ax1.plot(tlo,caplo,'-',c='k',label="caplo")
    ax2.plot(tmed,capmed,'-',c='k',label="capmed")
    ax3.plot(thigh,caphigh,'-',c='k',label="caphigh")
    fig.text(0.05, .5 , 'capacitance '+r'$(\Delta\%)$', ha='center', va='center', rotation='vertical')
    ax1.set_ylim(cmin-abs(cmin)*.15,cmax)
    #ax1.set_xlim(.9,np.max(start_en)*1.15)
    ax2.set_ylim(cmin-abs(cmin)*.15,cmax)
    #ax2.set_xlim(.9,np.max(load_en)*1.15)
    ax3.set_ylim(cmin-abs(cmin)*.15,cmax)
    #ax3.set_xlim(.9,np.max(load_en)*1.15)
    ax1.ticklabel_format(axis='y',style='plain', scilimits=(0,0)) #,style='sci'
    ax1.set_xticks([])
    ax2.ticklabel_format(axis='y',style='plain', scilimits=(0,0)) #,style='sci'
    ax2.set_xticks([])
    ax3.ticklabel_format(axis='y',style='plain', scilimits=(0,0)) #,style='sci'
    ax3.set_xticks([])
    fig.text(.55, 0.025, 'time', ha='center', va='center')
    fig.text(0.09, 0.955, 'a)', ha='center', va='center')
    fig.text(0.09, 0.64, 'b)', ha='center', va='center')
    fig.text(0.09, 0.32, 'c)', ha='center', va='center')
    #fig.text(0.1, 0.03, 'c)', ha='center', va='center')
    plt.tight_layout(rect=(.09,.01,.99,.99))
    plt.savefig('cappicsampv.png',dpi=300)


if plot_picsamp:
    plt.rcParams.update({'font.size': 12})
    plt.close('all')
    pic_samples=['13','19','18']
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=len(pic_samples), ncols=1,figsize=(5.82677*2/3,6))
    tlo,caplo=np.linspace(0, 12,num=len(CAP_REGIION[pic_samples[0]][:,1])),CAP_REGIION[pic_samples[0]][:,1]*100
    tmed,capmed=np.linspace(0, 12,num=len(CAP_REGIION[pic_samples[1]][:,1])),CAP_REGIION[pic_samples[1]][:,1]*100
    thigh,caphigh=np.linspace(0, 12,num=len(CAP_REGIION[pic_samples[2]][:,1])),CAP_REGIION[pic_samples[2]][:,1]*100
    cmax=np.max([np.max(caplo),np.max(capmed),np.max(caphigh)])*.5
    cmin=np.min([np.min(caplo),np.min(capmed),np.min(caphigh)])*1
    ax1.plot(tlo,caplo,'.',c='k',label="caplo")
    ax2.plot(tmed,capmed,'.',c='k',label="capmed")
    ax3.plot(thigh,caphigh,'.',c='k',label="caphigh")
    fig.text(0.05, .5 , 'capacitance '+r'$(\Delta\%)$', ha='center', va='center', rotation='vertical')
    ax1.set_ylim(cmin-abs(cmin)*.15,cmax)
    #ax1.set_xlim(.9,np.max(start_en)*1.15)
    ax2.set_ylim(cmin-abs(cmin)*.15,cmax)
    #ax2.set_xlim(.9,np.max(load_en)*1.15)
    ax3.set_ylim(cmin-abs(cmin)*.15,cmax)
    #ax3.set_xlim(.9,np.max(load_en)*1.15)
    #ax1.ticklabel_format(axis='y',style='plain', scilimits=(0,0)) #,style='sci'
    ax1.set_xticks([])
    #ax2.ticklabel_format(axis='y',style='plain', scilimits=(0,0)) #,style='sci'
    ax2.set_xticks([])
    #ax3.ticklabel_format(axis='y',style='plain', scilimits=(0,0)) #,style='sci'
    #ax3.set_xticks([])
    fig.text(.55, 0.025, 'time (s)', ha='center', va='center')
    fig.text(0.09, 0.96, 'a)', ha='center', va='center')
    fig.text(0.09, 0.66, 'b)', ha='center', va='center')
    fig.text(0.09, 0.35, 'c)', ha='center', va='center')
    #fig.text(0.1, 0.03, 'c)', ha='center', va='center')
    plt.tight_layout(rect=(.09,.01,.99,.99))
    plt.savefig('cappicsampscatv.png',dpi=300)





linestr=('-'*40)+'\n'
print(linestr+'key table\n'+linestr)

export=[[DCOC[key],START_en[key],LOAD_work[key],key] for key in skey]
export.sort(key=lambda x:x[2])
ex=''
for line in export:
    ex+=str(round(line[2],6))+' & '+str(round(line[1],6))+' & '+str(round(line[0],11))+'\\\\'+'\n'
print(ex)

linestr=('-'*40)+'\n'
print(linestr+'key table\n'+linestr)

ex=''
for line in export:
    ex+=str(line[3])+' --> '+str(round(line[2],6))+'\n'
print(ex)




#%% generate learing curves for a linear model

plot_dual = True
if plot_dual:
    C=np.asarray([DCOC[key]/2 for key in skey ])
    start_en=np.asarray([START_en[key] for key in skey ])
    load_en=np.asarray([LOAD_work[key] for key in skey ])
    xmodel = np.linspace(0,max(load_en)*2,num=100)
    m, b = np.polyfit(load_en,C,deg=1)
    Ym = [max(m*k+b,0) for k in xmodel] 
    
    
    up=5.084321175
    med=2.881115333
    x0, x1 = np.meshgrid(np.linspace(0,start_en[-1]*2,800),np.linspace(0,1,2))
    z = x0*0
    for ix,iy in np.ndindex(x0.shape):
        if x0[ix,iy]>up:
            z[ix,iy]=10
        elif x0[ix,iy]<=up and x0[ix,iy]>=med:
            z[ix,iy]=5
        else:
            z[ix,iy]=0
    legend_elements = [Patch(facecolor='bisque', edgecolor='k',label='safe',alpha=.5),
                       Patch(facecolor='gold', edgecolor='k',label='marginal',alpha=.5),
                       Patch(facecolor='tab:orange', edgecolor='k',label='unsafe',alpha=.5)]
    fig, ax1 = plt.subplots(nrows=1, ncols=1,figsize=(6,4))
    # fig.text(0.3, 0.94, 'energy absorbed', ha='center', va='center')
    # ax1.plot(start_en,C,'o',c='k',label="cap")
    # ax1.contourf(x0,x1,z,levels=3,colors=['bisque','gold','tab:orange','tab:orange'],alpha=.5)
    #ax2.plot(1+(load_en-nompr)/nompr,C,'o',c='tab:green',label="cap")
    ax1.plot(load_en,C,'o',c='k',label="cap")
    ax1.plot(xmodel,Ym,'k-.',label='model')
    ax1.contourf(x0,x1,z,levels=3,colors=['bisque','gold','tab:orange','tab:orange'],alpha=.5)
    fig.text(0.5, 0.05, 'energy absorbed by the plate (J)', ha='center', va='center')
    fig.text(0.03, 0.55, 'capacitance '+r'$(\Delta\%)$', ha='center', va='center', rotation='vertical')
    # ax1.set_ylim(0,np.max(C)*1.15)
    # ax1.set_xlim(.9,np.max(start_en)*1.15)
    ax1.set_ylim(0,np.max(C)*1.15)
    ax1.set_xlim(.9,np.max(load_en)*1.15)
    ax1.ticklabel_format(axis='y',style='plain', scilimits=(0,0)) #,style='sci'
    ax1.legend(handles=legend_elements, loc=2,framealpha=1)
    plt.tight_layout(rect=(.05,.05,.95,.95))
    plt.savefig('uncombinedfig_fit.png',dpi=400)
    

plot_dual = True
if plot_dual:
    C=np.asarray([DCOC[key]/2 for key in skey ])
    start_en=np.asarray([START_en[key] for key in skey ])
    load_en=np.asarray([LOAD_work[key] for key in skey ])
    xmodel = np.linspace(0,max(load_en)*2,num=100)
    m, b = np.polyfit(load_en,C,deg=1)
    Ym = [max(m*k+b,0) for k in xmodel] 
    print(m)
    print(b)
    up=5.084321175
    med=2.881115333
    x0, x1 = np.meshgrid(np.linspace(0,start_en[-1]*2,800),np.linspace(0,1,2))
    z = x0*0
    for ix,iy in np.ndindex(x0.shape):
        if x0[ix,iy]>up:
            z[ix,iy]=10
        elif x0[ix,iy]<=up and x0[ix,iy]>=med:
            z[ix,iy]=5
        else:
            z[ix,iy]=0
    legend_elements = [Patch(facecolor='bisque', edgecolor='k',label='safe',alpha=.5),
                       Patch(facecolor='gold', edgecolor='k',label='marginal',alpha=.5),
                       Patch(facecolor='tab:orange', edgecolor='k',label='unsafe',alpha=.5)]
    fig, ax1 = plt.subplots(nrows=1, ncols=1,figsize=(6,4))
    # fig.text(0.3, 0.94, 'energy absorbed', ha='center', va='center')
    # ax1.plot(start_en,C,'o',c='k',label="cap")
    # ax1.contourf(x0,x1,z,levels=3,colors=['bisque','gold','tab:orange','tab:orange'],alpha=.5)
    #ax2.plot(1+(load_en-nompr)/nompr,C,'o',c='tab:green',label="cap")
    ax1.plot(load_en,C,'o',c='k',label="cap")
    ax1.plot(xmodel,Ym,'k-.',label='model')
    ax1.contourf(x0,x1,z,levels=3,colors=['bisque','gold','tab:orange','tab:orange'],alpha=.5)
    fig.text(0.5, 0.05, 'energy absorbed by the plate (J)', ha='center', va='center')
    fig.text(0.03, 0.55, 'capacitance '+r'$(\Delta\%)$', ha='center', va='center', rotation='vertical')
    # ax1.set_ylim(0,np.max(C)*1.15)
    # ax1.set_xlim(.9,np.max(start_en)*1.15)
    ax1.set_ylim(0,0.0006)
    ax1.set_xlim(1.1,2.881115333*1.15)
    ax1.ticklabel_format(axis='y',style='plain', scilimits=(0,0)) #,style='sci'
    ax1.legend(handles=legend_elements, loc=2,framealpha=1)
    plt.tight_layout(rect=(.05,.05,.95,.95))
    plt.savefig('uncombinedfig_fit_mar.png',dpi=400)

plot_dual = True
if plot_dual:
    C=np.asarray([DCOC[key]/2 for key in skey ])
    start_en=np.asarray([START_en[key] for key in skey ])
    load_en=np.asarray([LOAD_work[key] for key in skey ])
    xmodel = np.linspace(0,max(load_en)*2,num=100)
    m, b = np.polyfit(load_en,C,deg=1)
    Ym = [max(m*k+b,0) for k in xmodel] 
    
    
    up=5.084321175
    med=2.881115333
    x0, x1 = np.meshgrid(np.linspace(0,start_en[-1]*2,800),np.linspace(0,1,2))
    z = x0*0
    for ix,iy in np.ndindex(x0.shape):
        if x0[ix,iy]>up:
            z[ix,iy]=10
        elif x0[ix,iy]<=up and x0[ix,iy]>=med:
            z[ix,iy]=5
        else:
            z[ix,iy]=0
    legend_elements = [Patch(facecolor='bisque', edgecolor='k',label='safe',alpha=.5),
                       Patch(facecolor='gold', edgecolor='k',label='marginal',alpha=.5),
                       Patch(facecolor='tab:orange', edgecolor='k',label='unsafe',alpha=.5)]
    fig, ax1 = plt.subplots(nrows=1, ncols=1,figsize=(6,4))
    # fig.text(0.3, 0.94, 'energy absorbed', ha='center', va='center')
    # ax1.plot(start_en,C,'o',c='k',label="cap")
    # ax1.contourf(x0,x1,z,levels=3,colors=['bisque','gold','tab:orange','tab:orange'],alpha=.5)
    #ax2.plot(1+(load_en-nompr)/nompr,C,'o',c='tab:green',label="cap")
    ax1.plot(load_en,C,'o',c='k',label="cap")
    ax1.plot(xmodel,Ym,'k-.',label='model')
    ax1.contourf(x0,x1,z,levels=3,colors=['bisque','gold','tab:orange','tab:orange'],alpha=.5)
    fig.text(0.5, 0.05, 'energy absorbed by the plate (J)', ha='center', va='center')
    fig.text(0.03, 0.55, 'capacitance '+r'$(\Delta\%)$', ha='center', va='center', rotation='vertical')
    # ax1.set_ylim(0,np.max(C)*1.15)
    # ax1.set_xlim(.9,np.max(start_en)*1.15)
    ax1.set_ylim(0,np.max(C)*1.15)
    ax1.set_xlim(.9,np.max(load_en)*1.15)
    ax1.ticklabel_format(axis='y',style='plain', scilimits=(0,0)) #,style='sci'
    ax1.legend(handles=legend_elements, loc=2,framealpha=1)
    plt.tight_layout(rect=(.05,.05,.95,.95))
    plt.savefig('uncombinedfig_fit_mid.png',dpi=400)
    



        
