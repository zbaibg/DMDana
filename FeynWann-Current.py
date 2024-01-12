import pandas as pd
import numpy as np
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
            
jx_elec_tot['time']=time
jx_elec_tot['j(t)']=currentx
jy_elec_tot['time']=time
jy_elec_tot['j(t)']=currenty
jz_elec_tot['time']=time
jz_elec_tot['j(t)']=currentz


jx_elec_tot.to_csv('jx_elec_tot.out',sep='\t',index=False)
jy_elec_tot.to_csv('jy_elec_tot.out',sep=' ',index=False)
jz_elec_tot.to_csv('jz_elec_tot.out',sep=' ',index=False)