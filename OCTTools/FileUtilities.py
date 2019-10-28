# utilities functions.

import numpy as np
import xml.etree.ElementTree as ET

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