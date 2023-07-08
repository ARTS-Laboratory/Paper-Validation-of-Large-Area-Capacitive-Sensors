# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:13:33 2020

@author: alexv
"""
import numpy as np
import sys
from odbAccess import *
#from AbaqusConstants import *

OBD='POINTBEND_FFFF_NSIDE.odb'
STEPV=['Step-01','Step-02','Step-03','Step-1','Step-11','Step-12','Step-13','Step-2','Step-21','Step-22','Step-23','Step-3']
for step in STEPV:
    shell1 = openOdb(OBD, readOnly=False)
    stepToRead = shell1.steps[step]
    frameToRead = stepToRead.frames[1]
    odbResults = frameToRead.fieldOutputs['U']
    nodeData = odbResults
    
    Label = []
    Displacements = []
    
    text = 'header'
    for node in nodeData.values:
        #Label = Label.append(node.nodeLabel)
        #Displacements = Displacements.append(node.data)
        line = str(node.nodeLabel)+'\t'+str(node.data)
        text = '\n'.join([text,line])
    
    File = file(step+'_results.txt', 'w')
    File.write(text)
    File.close()
    
    shell1.close()



    