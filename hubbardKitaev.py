import numpy as np
import scipy as sp
import sys, os
from tenpy.tools.params import asConfig
from tenpy.models.model import CouplingModel, MPOModel
from mpomps_tenpy import Eletron
from tenpy.networks.site import SpinSite, SpinHalfFermionSite
from tenpy.models.lattice import Lattice, Chain, Honeycomb
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

class Kitaev(CouplingModel):
    def __init__(self, model_params):
        model_params = asConfig(model_params, self.__class__.__name__)
        self.model_params = model_params
        Lx = model_params.get('Lx', 12)
        Ly = model_params.get('Ly', 4)
        self.Lx = Lx
        bc = model_params.get('bc', 'periodic')
        bc_MPS = model_params.get('bc_MPS', 'finite')
        cons_S = None
        self.cons_S = cons_S
        self.cons_N = model_params.get('cons_N', 'N')

        site = Eletron(cons_N=self.cons_N, cons_S=cons_S)
        # site = SpinSite(conserve="parity")
        # site.iSy = (site.Sm - site.Sp) * 0.5
        # site.opnames.add('iSy')

        self.site = site
        self.lat = Honeycomb(Lx, Ly, site, bc=['open', bc], bc_MPS=bc_MPS)
        CouplingModel.__init__(self, self.lat, explicit_plus_hc=False)

        self.init_terms(model_params)

        H_MPO = self.calc_H_MPO()
        if model_params.get('sort_mpo_legs', False):
            H_MPO.sort_legcharges()
        MPOModel.__init__(self, self.lat, H_MPO)

        homepath  = os.getcwd()
        if os.path.isdir(homepath+'/data/') == False:
            os.mkdir(homepath+'/data/')
        self.path = homepath + '/data/' + "KitaevS_Ly_{}_Lx_{}_Kxyz_{}_{}_{}/".format(Ly, Lx, self.Kx, self.Ky, self.Kz)
        if os.path.isdir(self.path) == False:
            os.mkdir(self.path)
           
    def init_terms(self, model_params):
        self.Kx = model_params.get('tx', 1.0)
        self.Ky = model_params.get('ty', 1.0)
        self.Kz = model_params.get('tz', 1.0)

        nx, ny = self.lat.Ls
        for _x in range(nx):
            for _y in range(ny):
                idxA = self.coordinate2lable(_x, _y, 0)
                idxB = self.coordinate2lable(_x, _y, 1)
                idzA = self.coordinate2lable(_x, _y+1, 0)
                idyA = self.coordinate2lable(_x+1, _y, 0)
                self.add_coupling_term(self.Kx, idxA, idxB, "Sx", "Sx", op_string='Id', plus_hc=False,)
                if _x < nx-1:
                    self.add_coupling_term(-self.Ky, idxB, idyA, "iSy", "iSy", op_string='Id', plus_hc=False,)
                if _y < ny-1:
                    self.add_coupling_term(self.Kz, idxB, idzA, "Sz", "Sz", op_string='Id', plus_hc=False,)
                else:
                    self.add_coupling_term(self.Kz, idzA, idxB, "Sz", "Sz", op_string='Id', plus_hc=False,)

    def plot_lat(self, ax=None):
        if ax is None:
            ax = plt.subplot(111)
        self.lat.plot_basis(ax)
        self.lat.plot_coupling(ax)
        self.lat.plot_bc_identified(ax)
        self.lat.plot_sites(ax)
        plt.show()

    def coordinate2lable(self, ix, iy, ab):
        Lx, Ly = self.lat.Ls
        # if self.model_params['order'] == "Cstyle":
        if self.model_params['bc_MPS'] == "infinite":
            unit_cell = ix  * Ly + iy
        else:
            unit_cell = (ix % Lx) * Ly + (iy % Ly)
        return unit_cell*2+ab
        # else:
            # raise "Now the order can only be Cstyle (no snake, no default, no....)"

    def plaquette(self, ix, iy):
        sites = []
        # if self.model_params['order'] == "Cstyle":
        sites.append( self.coordinate2lable(ix, iy, 0) )
        sites.append( self.coordinate2lable(ix, iy, 1) )
        sites.append( self.coordinate2lable(ix, iy+1, 0) )
        sites.append( self.coordinate2lable(ix-1, iy+1, 1) )
        sites.append( self.coordinate2lable(ix-1, iy+1, 0) )
        sites.append( self.coordinate2lable(ix-1, iy, 1) )
        return sites
        # else:
            # raise "Now the order can only be Cstyle (no snake, no default, no....)"

    def print_Z2flux(self, psi):
        nx, ny = self.lat.Ls
        N = nx*ny*2
        for column in range(nx):
            for row in range(ny):
                ops = ['Id'] * N
                i0, i1, i2, i3, i4, i5 = self.plaquette(column, row)
                ops[i0] = 'Sz'
                ops[i1] = 'Sy'
                ops[i2] = 'Sx'
                ops[i3] = 'Sz'
                ops[i4] = 'Sy'
                ops[i5] = 'Sx'
                print("Z2 flux at ix = {}, iy =      {}".format(column, row), 
                      psi.expectation_value_multi_sites(ops, 0)*64 )

    def print_wilsonloop(self, psi):
        nx, ny = self.lat.Ls
        N = nx*ny*2
        for _x in range(nx):
            ops = ['Id'] * N
            for _y in range(ny):
                ops[(_x*ny+_y)*2+0] = "Sy"            
                ops[(_x*ny+_y)*2+1] = "Sy"            
            print("wloop, ix = {}".format(_x), psi.expectation_value_multi_sites(ops, 0)*(2**(2*ny)) )
        for _y in range(ny):
            ops = ['Id'] * N
            for _x in range(nx):
                ops[(_x*ny+_y)*2+0] = "Sz"            
                ops[(_x*ny+_y)*2+1] = "Sz"            
            print("wloop, iy = {}".format(_y), psi.expectation_value_multi_sites(ops, 0)*(2**(2*nx)))

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
            # print("init total particle number", psiinit.expectation_value('Ntot').sum() )
            print("init total Sz number", psiinit.expectation_value('Sz').sum() )
        elif isinstance(init, str):
            with open (init, 'rb') as f:
                psiinit = pickle.load(f)
            dmrg_params['mixer'] = False
        elif isinstance(init, list):
            psiinit = MPS.from_product_state(self.lat.mps_sites(), init)            
        elif isinstance(init, MPS):
            psiinit = init
        else:
            print("wrong init")

        eng = dmrg.TwoSiteDMRGEngine(psiinit, self, dmrg_params)
        E, psidmrg = eng.run()
        print(E)
        psidmrg.canonical_form()

        self.psidmrg = psidmrg
        return psidmrg, E

class HubbardKitaev(CouplingModel):
    def __init__(self, model_params):
        model_params = asConfig(model_params, self.__class__.__name__)
        self.model_params = model_params
        Lx = model_params.get('Lx', 12)
        Ly = model_params.get('Ly', 4)
        self.Lx = Lx
        self.Ly = Ly
        self.Nlat = Lx*Ly*2
        bc = model_params.get('bc', 'periodic')
        bc_MPS = model_params.get('bc_MPS', 'finite')
        cons_S = None
        self.cons_S = cons_S

        site = Eletron(cons_N='N', cons_S=cons_S)
        self.site = site
        self.lat = Honeycomb(Lx, Ly, site, bc=['open', bc], bc_MPS=bc_MPS)
        CouplingModel.__init__(self, self.lat, explicit_plus_hc=False)

        self.init_terms(model_params)

        H_MPO = self.calc_H_MPO()
        if model_params.get('sort_mpo_legs', True):
            H_MPO.sort_legcharges()
        MPOModel.__init__(self, self.lat, H_MPO)
        model_params.warn_unused()

        homepath  = os.getcwd()
        if os.path.isdir(homepath+'/data/') == False:
            os.mkdir(homepath+'/data/')
        self.path = homepath + '/data/' + "KitaevH_half_filling_Ly_{}_Lx_{}_txyz_{}_{}_{}/".format(Ly, Lx, self.tx, self.ty, self.tz)
        if os.path.isdir(self.path) == False:
            os.mkdir(self.path)
           
    def init_terms(self, model_params):
        self.tx = model_params.get('tx', 1.0)
        self.ty = model_params.get('ty', 1.0)
        self.tz = model_params.get('tz', 1.0)
        self.t  = model_params.get('t',  1.0)
        self.U = model_params.get('U', 0.0)

        nx, ny = self.Lx, self.Ly
        x_bonds = []
        y_bonds = []
        z_bonds = []
        for _x in range(nx):
            for _y in range(ny):
                idxA = self.coordinate2lable(_x, _y, 0)
                idxB = self.coordinate2lable(_x, _y, 1)
                idzA = self.coordinate2lable(_x, _y+1, 0)
                idyA = self.coordinate2lable(_x+1, _y, 0)
                x_bonds.append( ( idxA, idxB) )
                self.add_coupling_term(-self.t, idxA, idxB, 'CduF', 'Cu', op_string='JW', plus_hc=True)
                self.add_coupling_term(-self.t, idxA, idxB, 'Cdd', 'FCd', op_string='JW', plus_hc=True)
                self.add_coupling_term(-self.tx, idxA, idxB, 'CduF', 'FCd', op_string='JW', plus_hc=True)
                self.add_coupling_term(-self.tx, idxA, idxB, 'Cdd', 'Cu', op_string='JW', plus_hc=True)
                if _x < nx-1:
                    y_bonds.append( ( idxB, idyA) )
                    self.add_coupling_term(   -self.t, idxB, idyA, 'CduF', 'Cu', op_string='JW', plus_hc=True)
                    self.add_coupling_term(   -self.t, idxB, idyA, 'Cdd', 'FCd', op_string='JW', plus_hc=True)
                    self.add_coupling_term(+1j*self.ty, idxB, idyA, 'CduF', 'FCd', op_string='JW', plus_hc=True)
                    self.add_coupling_term(-1j*self.ty, idxB, idyA, 'Cdd', 'Cu', op_string='JW', plus_hc=True)
                # else:
                #     print("y", idxB, idyA)
                #     self.add_coupling_term(   -self.t, idyA, idxB,  'CduF', 'Cu', op_string='JW', plus_hc=True)
                #     self.add_coupling_term(   -self.t, idyA, idxB,  'Cdd', 'FCd', op_string='JW', plus_hc=True)
                #     self.add_coupling_term(+1j*self.ty, idyA, idxB, 'CduF', 'FCd', op_string='JW', plus_hc=True)
                #     self.add_coupling_term(-1j*self.ty, idyA, idxB, 'Cdd', 'Cu', op_string='JW', plus_hc=True)
                if _y < ny-1:
                    z_bonds.append( ( idxB, idzA) )
                    self.add_coupling_term(-self.t-self.tz, idxB, idzA, 'CduF', 'Cu', op_string='JW', plus_hc=True)
                    self.add_coupling_term(-self.t+self.tz, idxB, idzA, 'Cdd', 'FCd', op_string='JW', plus_hc=True)
                else:
                    z_bonds.append( ( idzA, idxB) )
                    self.add_coupling_term(-self.t-self.tz, idzA, idxB, 'CduF', 'Cu', op_string='JW', plus_hc=True)
                    self.add_coupling_term(-self.t+self.tz, idzA, idxB, 'Cdd', 'FCd', op_string='JW', plus_hc=True)
        print("x-bonds :", x_bonds)
        print("y-bonds :", y_bonds)
        print("z-bonds :", z_bonds)

        self.add_onsite(self.U, 0, 'NuNd')
        self.add_onsite(self.U, 1, 'NuNd')

    def plot_lat(self, ax=None):
        if ax is None:
            ax = plt.subplot(111)
        self.lat.plot_basis(ax)
        self.lat.plot_coupling(ax)
        self.lat.plot_bc_identified(ax)
        self.lat.plot_sites(ax)
        plt.show()

    def coordinate2lable(self, ix, iy, ab):
        Lx, Ly = self.Lx, self.Ly
        unit_cell = (ix % Lx) * Ly + (iy % Ly)
        return unit_cell*2+ab
        # else:
            # raise "Now the order can only be Cstyle (no snake, no default, no....)"

    def plaquette(self, ix, iy):
        sites = []
        # if self.model_params['order'] == "Cstyle":
        sites.append( self.coordinate2lable(ix, iy, 0) )
        sites.append( self.coordinate2lable(ix, iy, 1) )
        sites.append( self.coordinate2lable(ix, iy+1, 0) )
        sites.append( self.coordinate2lable(ix-1, iy+1, 1) )
        sites.append( self.coordinate2lable(ix-1, iy+1, 0) )
        sites.append( self.coordinate2lable(ix-1, iy, 1) )
        return sites
        # else:
            # raise "Now the order can only be Cstyle (no snake, no default, no....)"

    def print_Z2flux(self, psi):
        nx, ny = self.lat.Ls
        N = nx*ny*2
        for column in range(nx):
            for row in range(ny):
                ops = ['Id'] * N
                i0, i1, i2, i3, i4, i5 = self.plaquette(column, row)
                ops[i0] = 'Sz'
                ops[i1] = 'Sy'
                ops[i2] = 'Sx'
                ops[i3] = 'Sz'
                ops[i4] = 'Sy'
                ops[i5] = 'Sx'
                print("Z2 flux at ix = {}, iy   =   {}".format(column, row), 
                      psi.expectation_value_multi_sites(ops, 0)*(2**(6)))

    def print_wilsonloop(self, psi):
        nx, ny = self.lat.Ls
        N = nx*ny*2
        for _x in range(nx):
            ops = ['Id'] * N
            for _y in range(ny):
                ops[(_x*ny+_y)*2+0] = "Sy"            
                ops[(_x*ny+_y)*2+1] = "Sy"            
            print("wloop, ix = {}".format(_x), psi.expectation_value_multi_sites(ops, 0)*(2**(2*ny)))
        for _y in range(ny):
            ops = ['Id'] * N
            for _x in range(nx):
                ops[(_x*ny+_y)*2+0] = "Sz"            
                ops[(_x*ny+_y)*2+1] = "Sz"            
            print("wloop, iy = {}".format(_y), psi.expectation_value_multi_sites(ops, 0)*(2**(2*nx)) )

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
        elif isinstance(init, MPS):
            psiinit = init
        else:
            print("wrong init")
        eng = dmrg.TwoSiteDMRGEngine(psiinit, self, dmrg_params)
        E, psidmrg = eng.run()
        print("Eng = ", E)
        self.psidmrg = psidmrg
        return psidmrg, E

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-Lx", type=int, default=12)
    parser.add_argument("-Ly", type=int, default=4)
    parser.add_argument("-U",  type=float, default=-0.2)
    parser.add_argument("-tx",  type=float, default=1.0)
    parser.add_argument("-ty",  type=float, default=1.0)
    parser.add_argument("-tz",  type=float, default=1.0)
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
                        tx=args.tx, ty=args.ty, tz=args.tz,
                        U=np.round(args.U,5))
    model = HubbardKitaev(model_params)
    if args.init == 'i':
        initPsi = None
    else:
        initPsi = args.init
    if args.ifdmrg == 'dmrg':
        psidmrg, E = model.run_dmrg(chi_max=args.chi, max_sweeps=args.Sweeps, init=initPsi)
        print(E)
        initname = args.init.split('/')[-1]
        fname = model.path+'/DMRG_Psi_{}_U_{}_D_{}'.format(initname, args.U, args.chi)        
        with open (fname, 'wb') as f:
            pickle.dump(psidmrg, f)
