import os
import numpy as np
import pandas as pd
from pathlib import Path
import DarkNews as dn
from DarkNews import GenLauncher
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import time
import sys
np.set_printoptions(threshold=sys.maxsize)



timestr = time.strftime("%Y%m%d-%H%M%S")

def first_run():
  print("Initialization...")
  list_of_experiments = ['dune_nd_fhc',
                          'microboone',
                          'minerva_le_fhc',
                          'miniboone_fhc',
                          'nutev_fhc']

  fig2 = plt.figure()
  ax = plt.subplot(111)
  m5_values=np.linspace(0.120,0.200,7)
  #common_kwargs = {'nu_projectile': pdg.numu, 'nu_upscattered': pdg.neutrino4, 'helicity': 'conserving', 'nuclear_target': C12}
  #BENCHMARCH A "Dark Seesaw solution to.. " 
  for exp in list_of_experiments:
    print("Running experiment", exp)
    decay_length = []
    for j in m5_values:
      
      print("Launching", j, "mass")
      gen_object = GenLauncher(Umu5=1e-3, UD5=1/np.sqrt(2), chi=0.0031, gD=2, mzprime=0.03, m4=0.080, m5=j, neval=1000, experiment=exp, HNLtype="majorana")
      df_1 = gen_object.run(loglevel="INFO")
      decay_length.append(df_1.attrs['N5_ctau0'])
    
      print("Plotting")
      
    ax.plot(m5_values, decay_length,label='Experiment = %s'%exp)
    ax.set_ylabel('Decay Length [cm]')
    ax.set_xlabel('m5 [GeV]')
    ax.legend(loc='upper right')
    fig2.savefig('./test.png')
  plt.close()
  print("GenLauncher finished")

def comp_decay_lengths3PM(**kwargs):
  print("Initialization...")
  
  print("Computing decay length")
  gen_object = GenLauncher(neval=1000, **kwargs)
  df = gen_object.run(loglevel="INFO")
  return df
  #print('The decay length of N5 using BP',BP,'is:', df.attrs['N5_ctau0'])

def dark_seesaw():
  '''
  N.B.: BP in ref. paper have both UDiL and UDiR (chiral), whislt genlauncher takes
  only UDi (L-R symmetric)
  '''

  #BPA from "A Dark Seesaw solution.."

  common_kwargs_A = {'Umu4':45.5e-8,'Umu5':0, 'UD4':0.244e-1, 'UD5':5.00e-1, 'chi':0.0031, 'alphaD':0.39,
                  'mzprime':1.25, 'm4':0.035, 'm5':0.120, 'epsilon2':4.6e-4, 'HNLtype':"majorana"}
  benchmark_A = 'A'
  comp_decay_lengths3PM(benchmark_A,**common_kwargs_A)

  #BPB from "A Dark Seesaw solution.."

  common_kwargs_B = {'Umu4':0.00361e-8,'Umu5':157e-8, 'UD4':0.371e-1,'UD5':5.57e-1, 'chi':0.0031, 'alphaD':0.32,
                  'mzprime':1.25, 'm4':0.074, 'm5':0.146,'epsilon2':4.6e-4, 'HNLtype':"majorana"}
  benchmark_B = 'B'
  comp_decay_lengths3PM(benchmark_B,**common_kwargs_B)

  #BPC from "A Dark Seesaw solution.."

  common_kwargs_C = {'Umu4': 0.000256e-8,'Umu5':51.1e-8,'UD4':0.143, 'UD5':5.19e-1, 'chi':0.0031, 'alphaD':0.76,
                  'mzprime':1.25, 'm4':0.062, 'm5':0.110,'epsilon2':4.6e-4, 'HNLtype':"majorana"}
  benchmark_C = 'C'
  comp_decay_lengths3PM(benchmark_C,**common_kwargs_C)

  #BPD from "A Dark Seesaw solution.."

  common_kwargs_D = {'Umu4':0,'Umu5':22.7e-8, 'UD4':0.554e-1, 'UD5':4.83e-1, 'chi':0.0031, 'alphaD':0.11,
                  'mzprime':1.25, 'm4':0.275, 'm5':0.346,'epsilon2':4.6e-4, 'HNLtype':"majorana"}
  benchmark_D = 'D'
  comp_decay_lengths3PM(benchmark_D,**common_kwargs_D)

def dot4(p1,p2):
  #print("printing momenta", p1,p2)
  #print("boh vediamo", p1[0,:])
  a = (p1[0,:])*(p2[0,:])-(p1[1,:])*(p2[1,:])-(p1[2,:])*(p2[2,:])-(p1[3,:])*(p2[3,:])
  #a = np.multiply((p1[0,:]**2),(p2[0,:]**2))-np.multiply((p1[1,:]**2),(p2[1,:]**2))-np.multiply((p1[2,:]**2),(p2[2,:]**2))-np.multiply((p1[3,:]**2),(p2[3,:]**2))
  #a = (p1[0,0]**2)*(p2[0,0]**2)-(p1[1,0]**2)*(p2[1,0]**2)-(p1[2,0]**2)*(p2[2,0]**2)-(p1[3,0]**2)*(p2[3,0]**2)
  #a = p1[:,0]**2*p2[:,0]**2-p1[:,1]**2*p2[:,1]**2-p1[:,2]**2*p2[:,2]**2-p1[:,3]**2*p2[:,3]**2
  #print("printing prudct", a)
  return a

def get_decay_length_in_lab(p, l_decay_proper_cm):
  M = np.sqrt(dot4(p.T, p.T))
  print("mass:", M)
  #M = np.sqrt(dot4(p, p))
  #for i in range(len(p[:,0])):
   # if ((p[i,0]**2 -  M[i]**2)<0):
    #  print("negative value in rad!!!!", (p[i,0]**2 -  M[i]**2))
  gammabeta = (np.sqrt(p[:,0]**2 -  M**2))/M
  print("l decay proper", l_decay_proper_cm)
  l_decay_lab=l_decay_proper_cm*gammabeta
  #print("printing decay length in lab",l_decay_lab)
  return l_decay_lab

def plot_graphs(p,decayl,model):
  fig2 = plt.figure()
  ax = plt.subplot(111)
  ax.plot( p[:,0], decayl,'.', color='darkred')
  ax.set_ylabel('Decay Length [cm]')
  ax.set_xlabel(r'$E_\nu/$GeV')
  ax.legend()
  fig2.savefig('./plots/%s_%s.png' % (timestr,model))
  plt.close()

def plot_bar(p,decayl,model):
  fig2 = plt.figure()
  ax = plt.subplot(111)
  ax.bar(p[:,0], decayl)
  ax.set_ylabel('Decay Length [cm]')
  ax.set_xlabel(r'$E_\nu/$GeV')
  ax.legend()
  fig2.savefig('./plots/%s_%s.png' % (timestr,model))
  plt.close()

def plot_isto(p,decayl,model):
  # and now histogram neutrino energy as an example
  fig, ax = dn.plot_tools.std_fig(figsize=(10,4))

  # unfortunately, the columns of the nhumpy array format are not specified and the correspondence is done by hand
  _=ax.hist(p[:,0], weights=decayl, bins=50, color='blue', histtype='step', label='numpy')

  ax.legend()
  ax.set_ylabel('Decay Lengths [cm]')
  ax.set_xlabel(r'$E_\nu/$GeV')

  fig.savefig('./plots/%s_%s.png' % (timestr,model))
  plt.close()

def threeone():
  '''Testing values given by Jaime'''
  # ---------------------------3+1------------------------------
  m4 = 0.2233 #GeV
  mzprime = 0.3487 #GeV
  vmu4_def = 3.6e-5
  ud4_def = 1.0/np.sqrt(2.)
  gD_def = 2.
  #umu4_def = np.sqrt(1.0e-12)
  epsilon_def = 8e-4
  umu4_f = lambda vmu4 : vmu4/np.sqrt(vmu4**2 + gD_def**2 * ud4_def**4)
  umu4_c = umu4_f(vmu4_def)

  common_kwargs = {'Umu4':umu4_c, 'UD4':ud4_def, 'mzprime':mzprime, 'm4':m4,
                     'epsilon2':epsilon_def, 'gD':gD_def, 'HNLtype':"dirac"}
  
  df = comp_decay_lengths3PM(**common_kwargs)

  #Compute decay lengths
  l_decay_proper_cm = df.attrs['N4_ctau0']
  pN = df.P_decay_N_parent.values

  l_decay_lab_cm = get_decay_length_in_lab(pN, l_decay_proper_cm)
  #print("printing p 3+1", pN)
  #plot_graphs(pN,l_decay_lab_cm)
  model = '3+1'
  #plot_isto(pN,l_decay_lab_cm,model)
  plot_graphs(pN,l_decay_lab_cm,model)
  #plot_bar(pN,l_decay_lab_cm,model)


def threetwo():
  # ---------------------------3+2------------------------------
  mzprime = 1.25 #GeV
  m5 = 0.186 #GeV
  m4 = 0.1285 #GeV
  ud4_def = 1.0/np.sqrt(2.)
  ud5_def = 1.0/np.sqrt(2.)
  gD_def = 2.
  epsilon_def = 1e-2

  v54 = gD_def * ud5_def * ud4_def
  vmu5_def = 2.56e-6
  #vmu5_def = gD_def * ud5_def * (umu4_def*ud4_def + umu5_def*ud5_def) / np.sqrt(1 - umu4_def**2 - umu5_def**2)
  
  umu4_c32=vmu5_def/np.sqrt(2*vmu5_def**2 + gD_def**2 * ud5_def**2 * (ud4_def + ud5_def)**2)
  umu5_c32=umu4_c32

  common_kwargs = {'Umu4':umu4_c32, 'Umu5':umu5_c32, 'UD4':ud4_def,
                    'UD5':ud5_def, 'mzprime':mzprime, 'm4':m4, 'm5':m5,
                     'epsilon2':epsilon_def, 'gD':gD_def, 'HNLtype':"dirac"}

  df = comp_decay_lengths3PM(**common_kwargs)

  #Compute decay lengths
  l_decay_proper_cm = df.attrs['N5_ctau0']
  pN = df.P_decay_N_parent.values

  l_decay_lab_cm = get_decay_length_in_lab(pN, l_decay_proper_cm)
  #print("printing printing p 3+2", pN)
  #plot_graphs(pN,l_decay_lab_cm)
  model = '3+2'
  #plot_isto(pN,l_decay_lab_cm,model)
  plot_graphs(pN,l_decay_lab_cm,model)
  #plot_bar(pN,l_decay_lab_cm,model)

if __name__=='__main__':

  #first_run()
  #dark_seesaw()

  threeone()
  threetwo()

  

      



  