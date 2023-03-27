import os
import numpy as np
import pandas as pd
from pathlib import Path
import DarkNews as dn
from DarkNews import GenLauncher
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

print("Initialization...")
list_of_experiments = [
'dune_nd_fhc',
'microboone',
'minerva_le_fhc',
'miniboone_fhc',
'nutev_fhc']
decay_length = []
m5_values=np.linspace(0.120,0.200,7)
#common_kwargs = {'nu_projectile': pdg.numu, 'nu_upscattered': pdg.neutrino4, 'helicity': 'conserving', 'nuclear_target': C12}
#BENCHMARCH A "Dark Seesaw solution to.. " 
for exp in list_of_experiments:
  for j in m5_values:
    print("Launching", j, "mass")
    gen_object = GenLauncher(Umu5=1e-3, UD5=1/np.sqrt(2), chi=0.0031, gD=2, mzprime=0.03, m4=0.080, m5=j, neval=1000, experiment=exp, HNLtype="majorana")
    df_1 = gen_object.run(loglevel="INFO")
    decay_length.append(df_1.attrs['N5_ctau0'])
  
  print("Plotting")
  fig2 = plt.figure()
  plt.plot(m5_values, decay_length,label=exp)
  plt.ylabel('Decay Length [cm]')
  plt.xlabel('m5 [GeV]')
  plt.savefig('./test.png')
  plt.close(fig2)
print("GenLauncher finished")


