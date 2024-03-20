#! /usr/bin/env python
import pandas as pd
import numpy as np
from ..lib.constant import *
from .global_variable import config
def do():
    jx_elec_tot=pd.DataFrame()
    jy_elec_tot=pd.DataFrame()
    jz_elec_tot=pd.DataFrame()
    time=[]
    currentx=[]
    currenty=[]
    currentz=[]
    with open('lindbladLinear.out') as f:
        for line in f.readlines():
            if 'Integrate: Step' in line:
                time.append(line.split()[4])
            if 'j: [' in line:
                currentx.append(line.split()[2])
                currenty.append(line.split()[3])
                currentz.append(line.split()[4])
    time=np.array(time,dtype=float)*fs
    currentx=np.array(currentx,dtype=float)/10000# turn A/m2 to A/cm2
    currenty=np.array(currenty,dtype=float)/10000# turn A/m2 to A/cm2
    currentz=np.array(currentz,dtype=float)/10000# turn A/m2 to A/cm2
    jx_elec_tot['time(a.u.)']=time
    jx_elec_tot['j(t)(A/cm^2)']=currentx
    jy_elec_tot['time']=time
    jy_elec_tot['j(t)(A/cm^2)']=currenty
    jz_elec_tot['time']=time
    jz_elec_tot['j(t)(A/cm^2)']=currentz


    jx_elec_tot.to_csv('jx_elec_tot.out',sep='\t',index=False)
    jy_elec_tot.to_csv('jy_elec_tot.out',sep='\t',index=False)
    jz_elec_tot.to_csv('jz_elec_tot.out',sep='\t',index=False)