# utilities functions.

import numpy as np
import xml.etree.ElementTree as ET
import os

def loadInputFilesList(filename):
    filesList = []
    with open( filename, "r") as f:
        for line in f:
            filesList.append(line.strip())
    return filesList

def saveInputFilesList(filesList, filename):
    with open( filename, "w") as f:
        for file in filesList:
            f.write(file + "\n")

def getSurfacesArray(segFile):
    """

    :param segFile in xml format
    :return: array in  [Slices, Surfaces,X] order
    """
    xmlTree = ET.parse(segFile)
    xmlTreeRoot = xmlTree.getroot()

    size = xmlTreeRoot.find('scan_characteristics/size')
    for item in size:
        if item.tag == 'x':
            X =int(item.text)
        elif item.tag == 'y':
            Y = int(item.text)
        elif item.tag == 'z':
            Z = int(item.text)
        else:
            continue

    surface_num = int(xmlTreeRoot.find('surface_num').text)
    surfacesArray = np.zeros((Z, surface_num, X))

    s = -1
    for surface in xmlTreeRoot:
        if surface.tag =='surface':
            s += 1
            z = -1
            for bscan in surface:
                if bscan.tag =='bscan':
                   z +=1
                   x = -1
                   for item in bscan:
                       if item.tag =='y':
                           x +=1
                           surfacesArray[z,s,x] = int(item.text)
    return surfacesArray

def getPatientID_Slice(fileName):
    '''

    :param fileName: e.g. "/home/hxie1/data/OCT_Beijing/control/4162_OD_23992_Volume/20110616012458_OCT10.jpg"
    :return: e.g. '4162_OD_23992_OCT10'
    '''
    splitPath = os.path.split(fileName)
    s = os.path.basename(splitPath[0])
    patientID = s[0:s.rfind("_")]
    s = splitPath[1]
    sliceID = s[s.rfind("_") + 1:s.rfind('.jpg')]
    return patientID, sliceID
