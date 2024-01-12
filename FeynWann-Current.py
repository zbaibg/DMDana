import pandas as pd
import numpy as np
fs  = 41.341373335# fs to a.u.
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
jx_elec_tot['time(a.u.)']=time
jx_elec_tot['j(t)(A/cm^2)']=currentx
jy_elec_tot['time']=time
jy_elec_tot['j(t)(A/cm^2)']=currenty
jz_elec_tot['time']=time
jz_elec_tot['j(t)(A/cm^2)']=currentz


jx_elec_tot.to_csv('jx_elec_tot.out',sep='\t',index=False)
jy_elec_tot.to_csv('jy_elec_tot.out',sep=' ',index=False)
jz_elec_tot.to_csv('jz_elec_tot.out',sep=' ',index=False)