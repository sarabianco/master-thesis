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
decay_length = []
m5_values=np.linspace(0.120,0.200,20)
for j in m5_values:
  print("Launching", j, "mass")
  gen_object = GenLauncher(Umu5=1e-3, UD5=1/np.sqrt(2), chi=0.0031, gD=2, mzprime=0.03, m4=0.080, m5=j, neval=1000, HNLtype="dirac")
  df_1 = gen_object.run(loglevel="INFO")
  decay_length.append(df_1.attrs['N5_ctau0'])

print("GenLauncher finished")


print("Plotting")

fig2 = plt.figure()
plt.plot(m5_values, decay_length)
plt.savefig('./test.png')
plt.close(fig2)