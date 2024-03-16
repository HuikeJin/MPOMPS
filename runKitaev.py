from PartonHoneycomb import *
from mpomps_tenpy import *
from hubbardKitaev import *
import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-Lx", type=int, default=12)
parser.add_argument("-Ly", type=int, default=4)
parser.add_argument("-bcy", type=int, default=1)
parser.add_argument("-P", type=float, default=-1)
parser.add_argument("-t", type=float, default=1.0)
parser.add_argument("-U", type=float, default=9.)
parser.add_argument("-chi",     type=int, default=100)
parser.add_argument("-chiF",     type=int, default=20)
parser.add_argument("-verbose", type=int, default=1)
parser.add_argument("-Sweeps",  type=int, default=6)
parser.add_argument("-init",    type=str, default='i')
parser.add_argument("-init2",    type=str, default='i')
parser.add_argument("-job", type=str, default='mpomps')
args = parser.parse_args()

import logging
logging.basicConfig(level=args.verbose)
for _ in ['parso.python.diff', 'parso.cache', 'parso.python.diff', 
          'parso.cache', 'matplotlib.font_manager', 'tenpy.tools.cache', 
          'tenpy.algorithms.mps_common', 'tenpy.linalg.lanczos', 'tenpy.tools.params']:
    logging.getLogger(_).disabled = True

if args.P < 0:
    args.P = np.inf

lx = args.Lx
ly = args.Ly
t = 1
params_spin = dict(lx=lx, ly=ly, bcy=args.bcy, bcx=0, Jx=1, Jy=1, Jz=1)
params_flux = dict(lx=lx, ly=ly, bcy=1, bcx=0, tx=t, ty=t, tz=t, tc=t, tf=0, uc=0, uf=0, P=args.P)

homepath  = os.getcwd()
if os.path.isdir(homepath+'/data/') == False:
    os.mkdir(homepath+'/data/')
# path = homepath + '/data/' + "Kitaev_half_filling_Ly_{}_Lx_{}/".format(ly, lx)
path = homepath + '/data/' + "KitaevH_half_filling_Ly_{}_Lx_{}_txyz_{}_{}_{}/".format(ly, lx, args.t, args.t, args.t)

if os.path.isdir(path) == False:
    os.mkdir(path)
if args.job == 'mpomps':    
    fname_spin = path+'/GMPOMPS_Kitaev_spinon_WY_{}_chi_{}'.format(args.bcy, args.chi)
    if os.path.isfile(fname_spin) == True:
        print("load the spinon wavefunction from ", fname_spin)
        with open (fname_spin, 'rb') as f:
            gpsi_spin = pickle.load(f)
    else:
        model_spin = HoneycombKitaev(params_spin)
        fu, fv = model_spin.calc_wannier_state()
        params_mpompsz2 = dict(cons_N="Z2", cons_S=None, trunc_params=dict(chi_max=args.chi))
        eng_spin = MPOMPSZ2(fu, fv, **params_mpompsz2)
        eng_spin.run()
        gpsi_spin = gutzwiller_projection(eng_spin.psi)    
        with open (fname_spin, 'wb') as f:
            pickle.dump(gpsi_spin, f)

    fname_flux = path+'/FMPOMPS_Kitaev_flux_P_{}_chi_{}'.format(args.P, args.chiF)
    if os.path.isfile(fname_flux) == True:
        print("load the flux wavefunction from ", fname_flux)
        with open (fname_flux, 'rb') as f:
            psi_flux = pickle.load(f)
    else:
        model_flux = FluxHoneycomb(params_flux)
        ffu = model_flux.calc_u_real()
        params_mpompsu1 = dict(cons_N="N", cons_S=None, trunc_params=dict(chi_max=args.chiF))
        eng_flux = MPOMPSU1(ffu, **params_mpompsu1)
        eng_flux.run()
        psi_flux = eng_flux.psi
        with open (fname_flux, 'wb') as f:
            pickle.dump(psi_flux, f)

    sp_psi = singlet_projection(psi_flux, gpsi_spin)    
    fname = path+'/SMPOMPS_Kitaev_Psi_P_{}_WY_{}_chiS_{}_chiF_{}'.format(args.P, args.bcy, args.chi, args.chiF)
    with open (fname, 'wb') as f:
        pickle.dump(sp_psi, f)
        
elif args.job == 'free':
    model_free = HoneycombSOC(params_flux)
    model_free.calc_wannier_state()
    uF = model_free.wannier_all 
    params_mpompsu1 = dict(cons_N="N", cons_S=None, trunc_params=dict(chi_max=args.chi))
    eng = MPOMPSU1(uF, **params_mpompsu1)
    eng.run()
    fname = path+'/FreeMPOMPS_Psi_chi_{}'.format(args.chi)
    with open (fname, 'wb') as f:
        pickle.dump(eng.psi, f)
elif args.job == 'dmrg':
    dmrg_params = dict(Ly=ly, Lx=lx, U=np.round(args.U,5), cons_S=None)
    dmrg_model = HubbardKitaev(dmrg_params)
    if args.init == 'i':
        psidmrg, E = dmrg_model.run_dmrg(chi_max=args.chi, max_sweeps=args.Sweeps, init=None)
    else:
        psidmrg, E = dmrg_model.run_dmrg(chi_max=args.chi, max_sweeps=args.Sweeps, init=args.init)
    print(E)
    dmrg_model.print_wilsonloop(psidmrg)
    dmrg_model.print_Z2flux(psidmrg)

    initname = args.init.split('/')[-1]
    fname = path+'/DMRG_Psi_{}_t_{}_U_{}_D_{}'.format(initname, t, args.U, args.chi)        
    with open (fname, 'wb') as f:
        pickle.dump(psidmrg, f)
elif args.job == 'ovlp':    
    with open (args.init, 'rb') as f:
        psi1 = pickle.load(f)
    with open (args.init2, 'rb') as f:
        psi2 = pickle.load(f)
    print("the overlap between ")
    print("---", args.init)
    print("---", args.init2)
    ovlp = psi1.overlap(psi2)
    print("is ",  abs(ovlp), ovlp)
elif args.job == 'eng':
    dmrg_params = dict(Ly=ly, Lx=lx, U=np.round(args.U,5), cons_S=None)
    dmrg_model = HubbardKitaev(dmrg_params)
    with open (args.init, 'rb') as f:
        psi1 = pickle.load(f)
    print( dmrg_model.H_MPO.expectation_value(psi1) )
    dmrg_model.print_wilsonloop(psi1)
    dmrg_model.print_Z2flux(psi1)
