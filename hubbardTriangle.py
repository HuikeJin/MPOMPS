import numpy as np
import scipy as sp
import sys, os
from tenpy.tools.params import asConfig
from tenpy.models.model import CouplingModel, MPOModel
from mpomps_tenpy import Eletron
from tenpy.models.lattice import Lattice, Chain, Triangular
from tenpy.algorithms import dmrg
from tenpy.algorithms.truncation import TruncationError
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
import tenpy.linalg.np_conserved as npc
import pickle
from matplotlib import pyplot as plt

def mps1_mps2(mps1, mps2):
    assert len(mps1._B) == len(mps2._B)
    L = len(mps1._B)
    left = npc.tensordot(mps1._B[0].conj(), mps2._B[0], axes=('p*', 'p'))
    for _ in range(1, L):
        left = npc.tensordot(left, mps1._B[_].conj(), axes=(['vR*'],["vL*"]))
        left = npc.tensordot(left, mps2._B[_], axes=(['vR','p*'],['vL','p']))
    value = left.to_ndarray()
    return value.reshape(-1)[0]

def mps1_mpo_mps2(mps1, mpo, mps2):
    assert len(mps1._B) == len(mpo) == len(mps2._B)
    L = len(mps1._B)
    temp = npc.tensordot(mps1._B[0].conj(), mpo[0], axes=('p*', 'p'))
    left = npc.tensordot(temp, mps2._B[0], axes=('p*', 'p'))
    for _ in range(1, L):
        temp = npc.tensordot(mps1._B[_].conj(), mpo[_], axes=('p*', 'p'))
        left = npc.tensordot(left, temp, axes=(['vR*', 'wR'],["vL*", 'wL']))
        left = npc.tensordot(left, mps2._B[_], axes=(['vR','p*'],['vL','p']))
    value = left.to_ndarray()
    return value.reshape(-1)[0]*mps1.norm*mps2.norm

class Hubbard(CouplingModel):
    def __init__(self, model_params):
        model_params = asConfig(model_params, self.__class__.__name__)
        self.model_params = model_params
        Lx = model_params.get('Lx', 12)
        Ly = model_params.get('Ly', 4)
        self.Lx = Lx
        bc = model_params.get('bc', 'periodic')
        bc_MPS = model_params.get('bc_MPS', 'finite')
        cons_S = model_params.get('cons_S', "2*Sz")
        self.cons_S = cons_S

        site = Eletron(cons_N='N', cons_S=cons_S)
        self.site = site
        self.lat = Triangular(Lx, Ly, site, bc=['open', bc], bc_MPS=bc_MPS)
        CouplingModel.__init__(self, self.lat, explicit_plus_hc=False)

        self.init_terms(model_params)

        H_MPO = self.calc_H_MPO()
        if model_params.get('sort_mpo_legs', False):
            H_MPO.sort_legcharges()
        MPOModel.__init__(self, self.lat, H_MPO)
        model_params.warn_unused()

        homepath  = os.getcwd()
        if os.path.isdir(homepath+'/data/') == False:
            os.mkdir(homepath+'/data/')
        self.path = homepath + '/data/' + "HubbardT_half_filling_{}_Ly_{}_Lx_{}/".format(cons_S, Ly, Lx) 
        if os.path.isdir(self.path) == False:
            os.mkdir(self.path)
           
    def init_terms(self, model_params):
        self.t = model_params.get('t', 1.0)
        self.U = model_params.get('U', 0.0)
        for _ in self.lat.pairs['nearest_neighbors']:
            u, v, dx = _
            print(self.t, u, v, dx)
            self.add_coupling(-self.t, u, 'Cdu', v, 'Cu', dx, op_string='JW', plus_hc=True)
            self.add_coupling(-self.t, u, 'Cdd', v, 'Cd', dx, op_string='JW', plus_hc=True)

        self.add_onsite(self.U, 0, 'NuNd')

    def plot_lat(self, ax=None):
        if ax is None:
            ax = plt.subplot(111)
        self.lat.plot_basis(ax)
        self.lat.plot_coupling(ax)
        self.lat.plot_bc_identified(ax)
        self.lat.plot_sites(ax)
        plt.show()

    def run_dmrg(self, **kwargs):
        mixer      = kwargs.get('mixer', True)
        chi_max    = kwargs.get('chi_max', 100)
        max_E_err  = kwargs.get('max_E_err', 1e-12)
        max_sweeps = kwargs.get('max_sweeps', 6)
        min_sweeps = kwargs.get('min_sweeps', min(3, max_sweeps) )
        dmrg_params = dict(mixer=mixer, 
                           trunc_params=dict(chi_max=chi_max),
                           max_E_err=max_E_err, 
                           max_sweeps=max_sweeps,
                           min_sweeps=min_sweeps)

        init = kwargs.get('init', None)
        if init is None:
            N = self.lat.N_sites
            fill = self.lat.N_sites//2
            init = [1]*(fill-1)+[2]*(fill-1)+[3]*(N-2*fill+1) + [0]
            np.random.shuffle(init)
            psiinit = MPS.from_product_state(self.lat.mps_sites(), init)
            for _ in range(min(chi_max,5)):
                np.random.shuffle(init)
                psiprime = MPS.from_product_state(self.lat.mps_sites(), init)
                psiinit = psiinit.add(psiprime, 1, 1)
                psiinit.canonical_form()
            psiinit.norm = 1
            psiinit.canonical_form()
            print("init total particle number", psiinit.expectation_value('Ntot').sum() )
            print("init total Sz number", psiinit.expectation_value('Sz').sum() )
        elif isinstance(init, str):
            with open (init, 'rb') as f:
                psiinit = pickle.load(f)
            dmrg_params['mixer'] = False
        elif isinstance(init, list):
            psiinit = MPS.from_product_state(self.lat.mps_sites(), init)            
        else:
            print("wrong init")

        eng = dmrg.TwoSiteDMRGEngine(psiinit, self, dmrg_params)
        E, psidmrg = eng.run()
        print(E)
        psidmrg.canonical_form()

        # fname = self.path+'/DMRG_Psi_t_{}_U_{}_D_{}'.format(self.t, self.U, chi_max)
        # with open (fname, 'wb') as f:
        #     pickle.dump(psidmrg, f)
        self.psidmrg = psidmrg
        return psidmrg, E

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-Lx", type=int, default=12)
    parser.add_argument("-Ly", type=int, default=4)
    parser.add_argument("-U",  type=float, default=-0.2)
    parser.add_argument("-chi",     type=int, default=5)
    parser.add_argument("-Sweeps",  type=int, default=5)
    parser.add_argument("-init",    type=str, default='i')
    parser.add_argument("-verbose", type=int, default=1)
    parser.add_argument("-cmprss", type=str, default="SVD")
    parser.add_argument("-ifdmrg", type=str, default='dmrg')
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=args.verbose)
    for _ in ['parso.python.diff', 'parso.cache', 'parso.python.diff', 
              'parso.cache', 'matplotlib.font_manager', 'tenpy.tools.cache', 
              'tenpy.algorithms.mps_common', 'tenpy.linalg.lanczos', 'tenpy.tools.params']:
        logging.getLogger(_).disabled = True

    model_params = dict(Ly=args.Ly, Lx=args.Lx,
                        U=np.round(args.U,5))
    model = Hubbard(model_params)
    if args.ifdmrg == 'dmrg':
        psidmrg, E = model.run_dmrg(chi_max=args.chi, max_sweeps=args.Sweeps)
    # if args.ifevol == 'evol':
    #     model.init_EvoEng(None, tmpo_params)
    #     model.run_Evo(args.Nsteps)
