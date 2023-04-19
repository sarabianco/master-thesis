import os
import numpy as np
import pandas as pd
from pathlib import Path
import DarkNews as dn
from DarkNews import GenLauncher
from DarkNews import const
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import time
import sys
from . import math_vecs as mv
from . import cuts
from . import exp_params as ep
#np.set_printoptions(threshold=sys.maxsize)
timestr = time.strftime("%Y%m%d-%H%M%S")

#---------------------DarkNews event generator launcher-----------------------

def comp_decay_lengths3PM(**kwargs):
  print("Initialization...")
  
  print("Computing decay length")
  gen_object = GenLauncher(neval=1000, **kwargs)
  df = gen_object.run(loglevel="INFO")
  return df

#------------------------Compute the complete spectum-------------------------

def compute_spectrum(df, EXP='miniboone', EVENT_TYPE='asymmetric'):
  """compute_spectrum _summary_
  Parameters
  ----------
  df : pd.DatagFrame
      DarkNews events (preferrably after selecting events inside detector)
  EXP : str, optional
      what experiment to use, by default 'miniboone' but can also be 'microboone'
  EVENT_TYPE : str, optional
      what kind of "mis-identificatin" selection to be used:
          for photons:
              'photon' assumes this is a photon and therefore always a single shower
          for lepton pairs:
              'asymmetric' picks events where one of the letpons (independent of charge) is below a hard threshold
              'overlapping' picks events where the two leptons are overlapping
              'both' for *either* asymmetric or overlapping condition to be true
              'separated' picks events where both letpons are above threshold and non-overlapping by default 'asymmetric'
  Returns
  -------
  pd.DatagFrame
      A new dataframe with additional columns containing weights of the selected events.
  """
  df = df.copy(deep=True)
  # Initial weigths
  w = df['w_event_rate'].values # typically already selected for fiducial volume

  if EVENT_TYPE=='photon':
      # Smear e+ and e-
      pgamma = cuts.smear_samples(df['P_decay_photon'], 0.0, EXP=EXP)

      # compute some reco kinematical variables from smeared electrons
      pgamma_mod = mv.modulus3(pgamma)
      costhetagamma = pgamma[3]/pgamma_mod
      Evis = pgamma[0]
      theta_beam = np.arccos(costhetagamma)*180.0/np.pi
  else:
      # Smear e+ and e-
      pep = cuts.smear_samples(df['P_decay_ell_plus'],const.m_e,EXP=EXP)
      pem = cuts.smear_samples(df['P_decay_ell_minus'],const.m_e,EXP=EXP)

      # compute some reco kinematical variables from smeared electrons
      pep_mod = mv.modulus3(pep)
      pem_mod = mv.modulus3(pem)
      costhetaep = pep[3]/pep_mod
      costhetaem = pem[3]/pem_mod
      Delta_costheta = mv.dot3(pem,pep)/pem_mod/pep_mod

      Evis, theta_beam, w, eff_s = signal_events(pep, pem, Delta_costheta, costhetaep, costhetaem, w, THRESHOLD=ep.THRESHOLD[EXP], ANGLE_MAX=ep.ANGLE_MAX[EXP], EVENT_TYPE=EVENT_TYPE)


  ############################################################################
  # Applies analysis cuts on the surviving LEE candidate events
  Evis2, theta_beam, w2, eff_c = expcuts(Evis, theta_beam, w, EXP=EXP,EVENT_TYPE=EVENT_TYPE)

  ############################################################################
  # Compute reconsructed neutrino energy
  # this assumes quasi-elastic scattering to mimmick MiniBooNE's assumption that the underlying events are nueCC.
  df['reco_Enu'] = const.m_proton * (Evis) / ( const.m_proton - (Evis)*(1.0 - np.cos(theta_beam)))

  eff_final, w2 = get_efficiencies(df['reco_Enu'],Evis,w,w2,EXP=EXP)

  ############################################################################
  # return reco observables of LEE -- regime is still a true quantity...
  df['reco_w'] = w2 * (6.1/5.)**3 # correcting for the fiducial volume cut already in MiniBooNE's efficiencies
  df['reco_Evis'] = Evis2
  df['reco_theta_beam'] = theta_beam
  df['reco_costheta_beam'] = np.cos(theta_beam*np.pi/180)
  df['reco_eff'] = eff_final

  return df

def get_efficiencies(reco_Enu, Evis, w, w2, EXP='miniboone'):
    if EXP=='miniboone':
        ###########################################################################
        # Now, a trick to get the approximate PMT and particle ID efficiencies
        # I am basically subtracting the event selection efficiency from the overall MiniBooNE-provided efficiency
        eff = np.array([0.0,0.089,0.135,0.139,0.131,0.123,0.116,0.106,0.102,0.095,0.089,0.082,0.073,0.067,0.052,0.026])
        enu = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.3,1.5,1.7,1.9,2.1])
        enu_c =  enu[:-1]+(enu[1:] - enu[:-1])/2
        eff_func = scipy.interpolate.interp1d(enu_c, eff, fill_value=(eff[0],eff[-1]), bounds_error=False, kind='nearest')
        wsum = w.sum()
        if wsum==0:
            eff_miniboone = (eff_func(reco_Enu)*w).sum()
        else:
            eff_miniboone = (eff_func(reco_Enu)*w).sum()/w.sum()

        ############################################################################
        # Now, apply efficiency as a function of energy to event weights
        w2 = eff_func(reco_Enu)*w2

        return eff_miniboone, w2

    elif EXP=='microboone':
        ############################################################################
        # Total nueCCQE efficiency -- using efficiencies provided by MicroBooNE
        E_numi_eff_edge = np.array([0,2.97415218e-1,4.70796471e-1,7.01196105e-1,9.88922361e-1,1.43183725e+0,3.00810009e+0,6.00428361e+0])
        nueCCQE_numi_eff_edge = np.array([6.13255034e-2,1.44127517e-1,2.12332215e-1,2.64681208e-1,2.76761745e-1,2.97902685e-1,2.57885906e-1,2.60151007e-1])
        E_numi_eff = (E_numi_eff_edge[1:] - E_numi_eff_edge[:-1])/2 + E_numi_eff_edge[:-1]
        nueCCQE_numi_eff = nueCCQE_numi_eff_edge[:-1]
        nueCCQE_numi_eff_func = scipy.interpolate.interp1d(E_numi_eff,nueCCQE_numi_eff,fill_value=(nueCCQE_numi_eff[0],nueCCQE_numi_eff[-1]),bounds_error=False,kind='nearest')

        muB_eff = (w*nueCCQE_numi_eff_func(Evis)).sum()/w.sum()

        w2 = nueCCQE_numi_eff_func(reco_Enu)*w2

        return muB_eff, w2

def signal_events(pep, pem, cosdelta_ee, costheta_ep, costheta_em, w, THRESHOLD=0.03, ANGLE_MAX=13.0, EVENT_TYPE='both'):
    """signal_events _summary_
  # This takes the events and asks for them to be either overlapping or asymmetric
        # THRESHOLD --
        # ANGLE_MAX --
    Parameters
    ----------
    pep : numpy.ndarray[ndim=2]
        four momenta of the positive lepton
    pem : numpy.ndarray[ndim=2]
        four momenta of the negative lepton
    cosdelta_ee : numpy.ndarray[ndim=1]
        cosine of the opening angle between lepton
    costheta_ep : numpy.ndarray[ndim=1]
        costheta of the positive lepton
    costheta_em : numpy.ndarray[ndim=1]
        costheta of the negative lepton
    w : numpy.ndarray[ndim=1]
        event weights
    
    THRESHOLD : float, optional
         how low energy does Esubleading need to be for event to be asymmetric, by default 0.03
    ANGLE_MAX : float, optional
         how wide opening angle needs to be in order to be overlapping, by default 13.0
    EVENT_TYPE : str, optional
        what kind of "mis-identificatin" selection to be used:
            'asymmetric' picks events where one of the letpons (independent of charge) is below a hard threshold
            'overlapping' picks events where the two leptons are overlapping
            'both' for *either* asymmetric or overlapping condition to be true
            'separated' picks events where both letpons are above threshold and non-overlapping by default 'asymmetric'
    Returns
    -------
    set of np.ndarrays
        Depending on the final event type, a list of energies and angles
    """
    ################### PROCESS KINEMATICS ##################
    # electron kinematics
    Eep = pep[0]
    Eem = pem[0]

    # angle of separation between ee
    theta_ee = np.arccos(cosdelta_ee)*180.0/np.pi

    # two individual angles
    theta_ep = np.arccos(costheta_ep)*180.0/np.pi
    theta_em = np.arccos(costheta_em)*180.0/np.pi

    # this is the angle of the combination of ee with the neutrino beam
    costheta_comb = (pem[3]+pep[3])/mv.modulus3(pem+pep)
    theta_comb = np.arccos(costheta_comb)*180.0/np.pi


    ########################################
    mee = np.sqrt(np.abs(mv.dot4(pep,pem)))
    mee_cut = 0.03203 + 0.007417*(Eep + Eem) + 0.02738*(Eep + Eem)**2
    inv_mass_cut = (mee < mee_cut)

    asym_p_filter = (Eem - const.m_e < THRESHOLD) & (Eep - const.m_e > THRESHOLD) & inv_mass_cut
    asym_m_filter = (Eem - const.m_e > THRESHOLD) & (Eep - const.m_e < THRESHOLD) & inv_mass_cut
    asym_filter = (asym_p_filter | asym_m_filter) & inv_mass_cut
    ovl_filter = (Eep - const.m_e > THRESHOLD) & (Eem - const.m_e > THRESHOLD) & (theta_ee < ANGLE_MAX) & inv_mass_cut
    sep_filter = (Eep - const.m_e > THRESHOLD) & (Eem - const.m_e > THRESHOLD) & (theta_ee > ANGLE_MAX) & inv_mass_cut
    inv_filter = (Eep - const.m_e < THRESHOLD) & (Eem - const.m_e < THRESHOLD) & inv_mass_cut
    both_filter = (asym_m_filter | asym_p_filter | ovl_filter)

    w_asym = w[asym_m_filter | asym_p_filter]
    w_ovl = w[ovl_filter]
    w_sep = w[sep_filter]
    w_inv = w[inv_filter]
    w_tot = w.sum()

    eff_asym	= w_asym.sum()/w_tot	
    eff_ovl		= w_ovl.sum()/w_tot	
    eff_sep		= w_sep.sum()/w_tot	
    eff_inv		= w_inv.sum()/w_tot	

    if EVENT_TYPE=='overlapping':
        ######################### FINAL OBSERVABLES ##########################################
        Evis = np.full_like(Eep, None)
        theta_beam = np.full_like(Eep, None)

        # visible energy
        Evis[ovl_filter] = (Eep*ovl_filter + Eem*ovl_filter)[ovl_filter]

        # angle to the beam
        theta_beam[ovl_filter] = theta_comb[ovl_filter]

        w[~ovl_filter] *= 0.0

        return Evis, theta_beam, w, eff_ovl

    elif EVENT_TYPE=='asymmetric':
        ######################### FINAL OBSERVABLES ##########################################
        Evis = np.full_like(Eep, None)
        theta_beam = np.full_like(Eep, None)

        # visible energy
        Evis[asym_filter] = (Eep*asym_p_filter + Eem*asym_m_filter)[asym_filter]

        # angle to the beam
        theta_beam[asym_filter] = (theta_ep*asym_p_filter + theta_em*asym_m_filter)[asym_filter]

        w[~asym_filter] *= 0.0

        return Evis, theta_beam, w, eff_asym

    elif EVENT_TYPE=='both':
        ######################### FINAL OBSERVABLES ##########################################
        Evis = np.full_like(Eep, None)
        theta_beam = np.full_like(Eep, None)
        
        # visible energy
        Evis[both_filter] = (Eep*asym_p_filter + Eem*asym_m_filter + (Eep+Eem)*ovl_filter)[both_filter]
        # angle to the beam
        theta_beam[both_filter] = (theta_ep*asym_p_filter + theta_em*asym_m_filter + theta_comb*ovl_filter)[both_filter]

        w[~both_filter] *= 0.0

        return Evis, theta_beam, w, eff_ovl+eff_asym

    elif EVENT_TYPE=='separated':
        ######################### FINAL OBSERVABLES ##########################################
        Eplus = np.full_like(Eep, None)
        Eminus = np.full_like(Eep, None)
        theta_beam_plus = np.full_like(Eep, None)
        theta_beam_minus = np.full_like(Eep, None)

        # visible energy
        Eplus[sep_filter] = Eep[sep_filter]
        Eminus[sep_filter] = Eem[sep_filter]

        # angle to the beam
        theta_beam_plus[sep_filter] = theta_ep[sep_filter]
        theta_beam_minus[sep_filter] = theta_em[sep_filter]
        theta_ee[sep_filter] = theta_ee[sep_filter]

        return Eplus, Eminus, theta_beam_plus, theta_beam_minus, theta_ee, w_sep, eff_sep
        
    elif EVENT_TYPE=='invisible':
        ######################### FINAL OBSERVABLES ##########################################
        # visible energy
        Eplus = np.full_like(Eep, None)
        Eminus = np.full_like(Eep, None)
        theta_beam_plus = np.full_like(Eep, None)
        theta_beam_minus = np.full_like(Eep, None)

        # visible energy
        Eplus[inv_filter] = Eep[inv_filter]
        Eminus[inv_filter] = Eem[inv_filter]

        # angle to the beam
        theta_beam_plus[inv_filter] = theta_ep[inv_filter]
        theta_beam_minus[inv_filter] = theta_em[inv_filter]
        theta_ee[inv_filter] = theta_ee[inv_filter]

        return Eplus, Eminus, theta_beam_plus, theta_beam_minus, theta_ee, w_inv, eff_inv

    else:
        print(f"Error! Could not find event type {EVENT_TYPE}.")
        return

def expcuts(Evis, theta_beam, w, EXP='miniboone',EVENT_TYPE='overlapping'):
    if EXP=='miniboone':
        return MB_expcuts(Evis, theta_beam, w)
    elif EXP=='microboone':
        return muB_expcuts(Evis, theta_beam, w, EVENT_TYPE=EVENT_TYPE)

def MB_expcuts(Evis, theta, weights):

	## Experimental cuts
	Pe  = np.sqrt(Evis**2 - const.m_e**2)
	Enu = (const.m_neutron*Evis - 0.5*const.m_e**2)/(const.m_neutron - Evis + Pe*np.cos(theta*np.pi/180.0))
	
	# Cuts
	in_energy_range = (Evis > ep.EVIS_MIN['miniboone']) & (Evis < ep.EVIS_MAX['miniboone'])
	# there could be an angular cuts, but miniboone's acceptance is assumed to be 4*pi
	final_selected = in_energy_range

	eff = weights[final_selected].sum()/weights.sum()

	Evis[~final_selected] = None
	theta[~final_selected] = None
	weights[~final_selected] *= 0.0

	return Evis, theta, weights, eff

def muB_expcuts(Evis, theta, weights, EVENT_TYPE='overlapping'):

	# Cuts
	in_energy_range = (Evis > ep.EVIS_MIN['microboone']) & (Evis < ep.EVIS_MAX['microboone'])
	
	final_selected = in_energy_range

	weights_fs = weights * final_selected

	eff = weights_fs.sum()/weights.sum()

	return Evis, theta, weights_fs, eff

#------------------------------Boost decay length-----------------------------

def get_decay_length_in_lab(p, l_decay_proper_cm):
  M = np.sqrt(dot4(p.T, p.T)) # mass
  gammabeta = (np.sqrt(p[:,0]**2 -  M**2))/M
  l_decay_lab=l_decay_proper_cm*gammabeta
  return l_decay_lab

#---------------------------------Plotting------------------------------------

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

def plot_isto(p,weight,model, title, xax, yax):
  # and now histogram neutrino energy as an example
  fig, ax = dn.plot_tools.std_fig(figsize=(10,4))

  # unfortunately, the columns of the nhumpy array format are not specified and the correspondence is done by hand
  _=ax.hist(p[:,0], weights=weight, bins=50, color='blue', histtype='step', label='numpy')

  ax.legend()
  ax.set_title(title)
  ax.set_ylabel(yax)
  ax.set_xlabel(xax)

  fig.savefig('./plots/%s_%s.png' % (timestr,model))
  plt.close()

# -----------------------------------3+1-------------------------------------

def threeone():
  '''Testing values given by Jaime'''
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
  return df

# -----------------------------------3+2-------------------------------------

def threetwo():

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
  return df

# ----------------------------- DECAY LENGTHS -------------------------------

def decay_length():
  '''3+1 model'''
  df1 = threeone()
  #Compute decay lengths
  l_decay_proper_cm = df1.attrs['N4_ctau0']
  pN = df1.P_decay_N_parent.values

  l_decay_lab_cm = get_decay_length_in_lab(pN, l_decay_proper_cm)

  model = '3+1'
  plot_graphs(pN,l_decay_lab_cm,model)

  '''3+2 model'''

  df2 = threetwo()
  #Compute decay lengths
  l_decay_proper_cm = df2.attrs['N5_ctau0']
  pN = df2.P_decay_N_parent.values

  l_decay_lab_cm = get_decay_length_in_lab(pN, l_decay_proper_cm)

  model = '3+2'
  plot_graphs(pN,l_decay_lab_cm,model)

# ------------------------- NUMBER OF EVENTS --------------------------------

def events():
  c=5

def events_filter():
  # Plot histograms for the 3+1 model
  df1=threeone()
  df3plus1 = compute_spectrum(df1)

  # Histogram reconstructed neutrino energy
  df3plus1filt = df3plus1[(df3plus1['reco_Enu']>0) & (df3plus1['reco_Enu']<2.5)]
  model1 = "3plus1"
  title1 = '3+1 model - Filtered'
  yax = 'Events'
  xaxE = r'Reconstructed $E_{\nu}/$GeV'
  plot_isto(df3plus1filt['reco_Enu'],df3plus1filt['w_event_rate',''],model1, title1, xaxE, yax)

  # Histogram theta angle
  xaxtheta=r'Reconstructed cos$\theta$'
  plot_isto(df3plus1['reco_costheta_beam'], df3plus1['w_event_rate',''],model1, title1, xaxtheta, yax)

  # Plot histograms for the 3+2 model
  df2=threetwo()
  df3plus2 = compute_spectrum(df2)
  df3plus2filt = df3plus2 [(df3plus2 ['reco_Enu']>0) & (df3plus2 ['reco_Enu']<2.5)]
  model2 = "3plus2"
  title2 = '3+2 model - Filtered'
  yax = 'Events'
  xaxE = r'Reconstructed $E_{\nu}/$GeV'
  plot_isto(df3plus2filt['reco_Enu'],df3plus2filt['w_event_rate',''],model2, title2, xaxE, yax)

  # Histogram theta angle
  xaxtheta=r'Reconstructed cos$\theta$'
  plot_isto(df3plus2['reco_costheta_beam'], df3plus2['w_event_rate',''],model2, title2, xaxtheta, yax)

if __name__=='__main__':
  decay_length()