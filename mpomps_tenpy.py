import matplotlib.pyplot as plt
from tenpy.tools.params import asConfig
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.networks.site import Site, SpinSite, SpinHalfFermionSite
from tenpy.models.lattice import Chain, Square
from tenpy.networks.mps import MPS
import tenpy.linalg.np_conserved as npc
import argparse
import pickle
import logging
logging.getLogger('parso.python.diff').disabled = True
logging.getLogger('parso.cache').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True
import matplotlib.pyplot as plt
import numpy as np

class Eletron(Site):
    r"""
    JW transformation:\n
    (F is spinful fermion, and C is spinful hardcore boson)

    Fd_{u,i} F_{u,j} = (Cd_{u,i} JW_i) JW_{i+1} ... JW_{j-2} JW_{j-1} C_{u,j} 

    Fd_{d,i} F_{d,j} = (Cd_{d,i} JWd_i) JW_{i+1} ... JW_{j-2} JW_{j-1} (JWu_{j} C_{d,j}) 
    = Cd_{d,i}  JW_{i+1} ... JW_{j-2} JW_{j-1} (JW_{j} C_{d,j})
    
    Fd_{u,i} F_{d,j} = (Cd_{u,i} JW_i) JW_{i+1} ... JW_{j-2} JW_{j-1} (JWu_{j} C_{d,j}) 
    = (Cd_{u,i} JW_i)  JW_{i+1} ... JW_{j-2} JW_{j-1} (JW_{j} C_{d,j})
    
    Fd_{d,i} F_{u,j} = (Cd_{d,i} JWd_i) JW_{i+1} ... JW_{j-2} JW_{j-1}  C_{u,j}
    = Cd_{d,i} JW_{i+1} ... JW_{j-2} JW_{j-1} C_{u,j}
    """
    def __init__(self, cons_N=None, cons_S=None):
        """
        the basis is {empty, up, down, double}
        """
        self.conserve = [cons_N, cons_S]
        self.cons_N = cons_N
        self.cons_S = cons_S
        if cons_N == 'N' and cons_S == '2*Sz':
            chinfo = npc.ChargeInfo([1, 1], ['N', '2*Sz'])
            leg = npc.LegCharge.from_qflat(chinfo, [[0, 0], [1, 1], [1, -1], [2, 0]])
        elif cons_N == 'N' and cons_S == 'parity':
            chinfo = chinfo = npc.ChargeInfo([1, 2], ['N', 'parity'])
            leg = npc.LegCharge.from_qflat(chinfo, [[0, 0], [1, 0], [1, 1], [2, 1]])
        elif cons_N == "N":
            chinfo = chinfo = npc.ChargeInfo([1], ['N'])
            leg = npc.LegCharge.from_qflat(chinfo, [0, 1, 1, 2])
        elif cons_N == "Z2" and cons_S == '2*Sz':
            chinfo = chinfo = npc.ChargeInfo([1, 1], ['N', '2*Sz'])
            leg = npc.LegCharge.from_qflat(chinfo, [[1, 0], [1, 1], [1, -1], [1, 0]])
        elif cons_N == "Z2" and cons_S == 'parity':
            chinfo = chinfo = npc.ChargeInfo([1, 2], ['N', 'parity'])
            leg = npc.LegCharge.from_qflat(chinfo, [[1, 0], [1, 0], [1, 1], [1, 1]])
        elif cons_N == 'Z2':
            chinfo = chinfo = npc.ChargeInfo([1], ['N'])
            leg = npc.LegCharge.from_qflat(chinfo, [1,1,1,1])
        else:
            leg = npc.LegCharge.from_trivial(4)

        JW = np.diag([1,  -1, -1,  1])
        JWu = np.diag([1, -1,  1, -1])
        JWd = np.diag([1,  1, -1, -1])
        Sp = np.array([[0.        , 0.        , 0.        , 0.],
                       [0.        , 0.        , 0.        , 0.],
                       [0.        , 1.        , 0.        , 0.],
                       [0.        , 0.        , 0.        , 0.]])
        Sm = Sp.T
        Sz = np.diag([0., 0.5, -0.5, 0.])
        Sx = (Sp + Sm)/2
        iSy = (Sp - Sm)/2
        Sy = 1j*(Sp - Sm)/2

        MZ = np.diag([-1., 0.0, 0.0, 1.0])
        MP = np.zeros((4,4)); MP[3, 0] = 1
        MM = np.zeros((4,4)); MM[0, 3] = 1
        MX = (MP + MM)
        MiY = (MP - MM)
        MY = 1j*(MP - MM)

        Cu = np.zeros((4, 4)); Cu[0, 1] = 1; Cu[2, 3] = 1;
        Cdu = Cu.T
        Cd = np.zeros((4, 4)); Cd[0, 2] = 1; Cd[1, 3] = 1;
        Cdd = Cd.T

        CduF = Cdu @ JW
        FCu  = CduF.T
        FCd  = JW @ Cd
        CddF = FCd.T
        
        Nh = np.diag([1., 0., 0., 0.])
        Nu = np.diag([0., 1., 0., 1.])
        Nd = np.diag([0., 0., 1., 1.])
        Ns = np.diag([0., 1., 1., 0.])
        Ntot = np.diag([0., 1., 1., 2.])
        NuNd = np.diag([0., 0., 0., 1.])

        if cons_S == "2*Sz":
            ops = dict(JW=JW, JWu=JWu, JWd=JWd,
                       Sp=Sp, Sm=Sm, Sz=Sz, 
                       MP=MP, MM=MM, MZ=MZ,
                       Cd=Cd, Cdd=Cdd, Cu=Cu, Cdu=Cdu,
                       CduF=CduF, FCu=FCu, FCd=FCd, CddF=CddF,
                       Nh=Nh, Nu=Nu, Nd=Nd, Ns=Ns, Ntot=Ntot, NuNd=NuNd)
        elif cons_N != "N":
            ops = dict(JW=JW, JWu=JWu, JWd=JWd,
                       Sp=Sp, Sm=Sm, Sz=Sz, 
                       Sx=Sx, Sy=Sy, iSy=iSy,
                       X=2*Sx, Y=2*Sy, Z=2*Sz,
                       MP=MP, MM=MM, MZ=MZ,
                       MX=MX, MY=MY, MiY=MiY,
                       Cd=Cd, Cdd=Cdd, Cu=Cu, Cdu=Cdu,
                       CduF=CduF, FCu=FCu, FCd=FCd, CddF=CddF,
                       Nh=Nh, Nu=Nu, Nd=Nd, Ns=Ns, Ntot=Ntot, NuNd=NuNd)
        else:
            ops = dict(JW=JW, JWu=JWu, JWd=JWd,
                       Sp=Sp, Sm=Sm, Sz=Sz, 
                       Sx=Sx, Sy=Sy, iSy=iSy,
                       X=2*Sx, Y=2*Sy, Z=2*Sz,
                       MP=MP, MM=MM, MZ=MZ,
                       Cd=Cd, Cdd=Cdd, Cu=Cu, Cdu=Cdu,
                       CduF=CduF, FCu=FCu, FCd=FCd, CddF=CddF,
                       Nh=Nh, Nu=Nu, Nd=Nd, Ns=Ns, Ntot=Ntot, NuNd=NuNd)
        names = ['empty', 'up', 'down', 'full']
        Site.__init__(self, leg, names, **ops)
        # Site.need_JW_string.update({'Cu','Cdu','Cd','Cdd'})

    def __repr__(self):
        """Debug representation of self."""
        return "site for hubbard model with conserve = {}".format(["N", self.cons_S])

class MPOMPSZ2():

    def __init__(self, u, v, **kwargs):
        self.cons_N = kwargs.get("cons_N", "Z2")
        self.cons_S = kwargs.get("cons_S", None)
        self.trunc_params = kwargs.get("trunc_params", dict(chi_max=4) )
        assert u.ndim == 2
        self._U = u
        self._V = v
        assert self._U.shape == self._V.shape
        self.projection_type = kwargs.get("projection_type", "Gutz")
        if self.projection_type == "Gutz":
            self.L = self.Llat = u.shape[1]//2 
        elif self.projection_type == "None":
            self.L = self.Llat = u.shape[1]//2 
        elif self.projection_type == "Ancilla":
            self.L = u.shape[1]//2 
            self.Llat = self.L//3
        self.site = Eletron(self.cons_N, self.cons_S)
        self.init_mps()

    def init_mps(self, init=None):
        L = self.L
        if init is None:
            init = [0] * L
        site = self.site
        self.init_psi = MPS.from_product_state([site]*L, init)
        self.psi = self.init_psi.copy()
        self.n_omode = 0
        return self.psi

    def calc_mpo_z2u1(self, u, v, s):
        '''
        if s = 0, u * c^dag_{up} + v c_{dn}, namely, with quamtum number 2*Sz = 1
        if s = 1, u * c^dag_{dn} + v c_{up}, namely, with quantum number 2*Sz = -1
        '''
        if s == 0:
            parity = [0, 1] 
        elif s == 1:
            parity = [0, -1]
        else:
            raise "s should be 0 (up) or 1 (dn)"

        chinfo = self.site.leg.chinfo
        pleg = self.site.leg
        ileg = npc.LegCharge.from_qflat(chinfo, [[0, 0]])
        bleg = npc.LegCharge.from_qflat(chinfo, [parity, [0, 0]])
        fleg = npc.LegCharge.from_qflat(chinfo, [parity])
        leg_frst = [ileg, bleg.conj(), pleg, pleg.conj()]
        leg_bulk = [bleg, bleg.conj(), pleg, pleg.conj()]
        leg_last = [bleg, fleg.conj(), pleg, pleg.conj()]

        mpo = []

        L = u.shape[0]//2
        mpo = []

        t0 = npc.zeros( leg_frst, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype )
        if s == 0:
            t0[0, 0, 1, 0] = u[0];  t0[0, 0, 0, 2] = v[1];
            t0[0, 0, 3, 2] = u[0];  t0[0, 0, 1, 3] = -v[1];  
        else:
            t0[0, 0, 2, 0] = u[1];  t0[0, 0, 0, 1] = v[0];
            t0[0, 0, 3, 1] = -u[1]; t0[0, 0, 2, 3] = v[0];
        t0[0, 1, 0, 0] = 1;
        t0[0, 1, 1, 1] = -1;
        t0[0, 1, 2, 2] = -1;
        t0[0, 1, 3, 3] = 1;
        mpo.append(t0)

        for i in range(1, L-1):
            ti = npc.zeros( leg_bulk, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype )
            ti[0, 0, 0, 0] = 1;
            ti[0, 0, 1, 1] = 1;
            ti[0, 0, 2, 2] = 1;
            ti[0, 0, 3, 3] = 1;
            if s == 0:
                ti[1, 0, 1, 0] = u[2*i+0];  ti[1, 0, 0, 2] = v[2*i+1];
                ti[1, 0, 3, 2] = u[2*i+0];  ti[1, 0, 1, 3] = -v[2*i+1];
            else:
                ti[1, 0, 2, 0] = u[2*i+1];  ti[1, 0, 0, 1] = v[2*i+0];
                ti[1, 0, 3, 1] = -u[2*i+1]; ti[1, 0, 2, 3] = v[2*i+0];
            ti[1, 1, 0, 0] = 1;
            ti[1, 1, 1, 1] = -1;
            ti[1, 1, 2, 2] = -1;
            ti[1, 1, 3, 3] = 1;
            mpo.append(ti)
        i = L-1
        tL = npc.zeros( leg_last, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype )
        tL[0, 0, 0, 0] = 1;
        tL[0, 0, 1, 1] = 1;
        tL[0, 0, 2, 2] = 1;
        tL[0, 0, 3, 3] = 1;
        if s == 0:
            tL[1, 0, 1, 0] = u[2*i+0];  tL[1, 0, 0, 2] = v[2*i+1];
            tL[1, 0, 3, 2] = u[2*i+0];  tL[1, 0, 1, 3] = -v[2*i+1]
        else:
            tL[1, 0, 2, 0] = u[2*i+1];  tL[1, 0, 0, 1] = v[2*i+0]  
            tL[1, 0, 3, 1] = -u[2*i+1]; tL[1, 0, 2, 3] = v[2*i+0];
        mpo.append(tL)
        return mpo

    def calc_mpo_z2parity(self, u, v, s):
        chinfo = self.site.leg.chinfo
        pleg = self.site.leg
        ileg = npc.LegCharge.from_qflat(chinfo, [[0, 0]])
        bleg = npc.LegCharge.from_qflat(chinfo, [[0, s], [0, 0]])
        fleg = npc.LegCharge.from_qflat(chinfo, [[0, s]])
        leg_frst = [ileg, bleg.conj(), pleg, pleg.conj()]
        leg_bulk = [bleg, bleg.conj(), pleg, pleg.conj()]
        leg_last = [bleg, fleg.conj(), pleg, pleg.conj()]

        mpo = []

        L = u.shape[0]//2
        mpo = []

        t0 = npc.zeros( leg_frst, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype )
        if s == 0:
            t0[0, 0, 1, 0] = u[0];  t0[0, 0, 0, 1] = v[0];
            t0[0, 0, 3, 2] = u[0];  t0[0, 0, 2, 3] = v[0];  
        else:
            t0[0, 0, 2, 0] = u[1];  t0[0, 0, 0, 2] = v[1];
            t0[0, 0, 3, 1] = -u[1]; t0[0, 0, 1, 3] = -v[1];
        t0[0, 1, 0, 0] = 1;
        t0[0, 1, 1, 1] = -1;
        t0[0, 1, 2, 2] = -1;
        t0[0, 1, 3, 3] = 1;
        mpo.append(t0)

        for i in range(1, L-1):
            ti = npc.zeros( leg_bulk, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype )
            ti[0, 0, 0, 0] = 1;
            ti[0, 0, 1, 1] = 1;
            ti[0, 0, 2, 2] = 1;
            ti[0, 0, 3, 3] = 1;
            if s == 0:
                ti[1, 0, 1, 0] = u[2*i+0];  ti[1, 0, 0, 1] = v[2*i+0];
                ti[1, 0, 3, 2] = u[2*i+0];  ti[1, 0, 2, 3] = v[2*i+0];
            else:
                ti[1, 0, 2, 0] = u[2*i+1];  ti[1, 0, 0, 2] = v[2*i+1];
                ti[1, 0, 3, 1] = -u[2*i+1]; ti[1, 0, 1, 3] = -v[2*i+1];
            ti[1, 1, 0, 0] = 1;
            ti[1, 1, 1, 1] = -1;
            ti[1, 1, 2, 2] = -1;
            ti[1, 1, 3, 3] = 1;
            mpo.append(ti)
        i = L-1
        tL = npc.zeros( leg_last, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype )
        tL[0, 0, 0, 0] = 1;
        tL[0, 0, 1, 1] = 1;
        tL[0, 0, 2, 2] = 1;
        tL[0, 0, 3, 3] = 1;
        if s == 0:
            tL[1, 0, 1, 0] = u[2*i+0];  tL[1, 0, 0, 1] = v[2*i+0]
            tL[1, 0, 3, 2] = u[2*i+0];  tL[1, 0, 2, 3] = v[2*i+0];
        else:
            tL[1, 0, 2, 0] = u[2*i+1];  tL[1, 0, 0, 2] = v[2*i+1];  
            tL[1, 0, 3, 1] = -u[2*i+1]; tL[1, 0, 1, 3] = -v[2*i+1]
        mpo.append(tL)
        return mpo

    def calc_mpo_z2(self, u, v):
        chinfo = self.site.leg.chinfo
        pleg = self.site.leg
        ileg = npc.LegCharge.from_qflat(chinfo, [[0]])
        bleg = npc.LegCharge.from_qflat(chinfo, [[0], [0]])
        fleg = npc.LegCharge.from_qflat(chinfo, [[0]])
        leg_frst = [ileg, bleg.conj(), pleg, pleg.conj()]
        leg_bulk = [bleg, bleg.conj(), pleg, pleg.conj()]
        leg_last = [bleg, fleg.conj(), pleg, pleg.conj()]

        mpo = []

        L = u.shape[0]//2
        mpo = []

        t0 = npc.zeros( leg_frst, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype )
        t0[0, 0, 1, 0] = u[0];  t0[0, 0, 0, 1] = v[0];
        t0[0, 0, 3, 2] = u[0];  t0[0, 0, 2, 3] = v[0];  
        t0[0, 0, 2, 0] = u[1];  t0[0, 0, 0, 2] = v[1];
        t0[0, 0, 3, 1] = -u[1]; t0[0, 0, 1, 3] = -v[1];
        t0[0, 1, 0, 0] = 1;
        t0[0, 1, 1, 1] = -1;
        t0[0, 1, 2, 2] = -1;
        t0[0, 1, 3, 3] = 1;
        mpo.append(t0)

        for i in range(1, L-1):
            ti = npc.zeros( leg_bulk, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype )
            ti[0, 0, 0, 0] = 1;
            ti[0, 0, 1, 1] = 1;
            ti[0, 0, 2, 2] = 1;
            ti[0, 0, 3, 3] = 1;
            ti[1, 0, 1, 0] = u[2*i+0];  ti[1, 0, 0, 1] = v[2*i+0];
            ti[1, 0, 3, 2] = u[2*i+0];  ti[1, 0, 2, 3] = v[2*i+0];
            ti[1, 0, 2, 0] = u[2*i+1];  ti[1, 0, 0, 2] = v[2*i+1];
            ti[1, 0, 3, 1] = -u[2*i+1]; ti[1, 0, 1, 3] = -v[2*i+1];
            ti[1, 1, 0, 0] = 1;
            ti[1, 1, 1, 1] = -1;
            ti[1, 1, 2, 2] = -1;
            ti[1, 1, 3, 3] = 1;
            mpo.append(ti)
        i = L-1
        tL = npc.zeros( leg_last, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype )
        tL[0, 0, 0, 0] = 1;
        tL[0, 0, 1, 1] = 1;
        tL[0, 0, 2, 2] = 1;
        tL[0, 0, 3, 3] = 1;
        tL[1, 0, 1, 0] = u[2*i+0];  tL[1, 0, 0, 1] = v[2*i+0]
        tL[1, 0, 3, 2] = u[2*i+0];  tL[1, 0, 2, 3] = v[2*i+0];
        tL[1, 0, 2, 0] = u[2*i+1];  tL[1, 0, 0, 2] = v[2*i+1];  
        tL[1, 0, 3, 1] = -u[2*i+1]; tL[1, 0, 1, 3] = -v[2*i+1]
        mpo.append(tL)
        return mpo

    def calc_mpo(self, u, v):
        if self.cons_S == "parity":
            s = 0;
            if abs(u[s::2]).sum() < 1e-12 and abs(v[s::2]).sum() < 1e-12:
                s = 1
            print("parity, s = ", s)
            return self.calc_mpo_z2parity(u, v, s)
        elif self.cons_S == '2*Sz':
            s = 0;
            if  abs(u[s::2]).sum() < 1e-12 and abs(v[s+1::2]).sum() < 1e-12:
                s = 1
            print("u1, s = ", s)
            return self.calc_mpo_z2u1(u, v, s)
        else:
            return self.calc_mpo_z2(u, v)

    def mpomps_step(self, u, v=None):
        if isinstance(u, int) and v is None:
            u = self._U[u, :]
            v = self._V[u, :]
        mpo = self.calc_mpo(u, v)
        mps = self.psi
        for i in range( self.L ):
            B = npc.tensordot(mps.get_B(i, 'B'), mpo[i], axes=('p', 'p*'))
            B = B.combine_legs([['wL', 'vL'], ['wR', 'vR']], qconj=[+1, -1])
            B.ireplace_labels(['(wL.vL)', '(wR.vR)'], ['vL', 'vR'])
            B.legs[B.get_leg_index('vL')] = B.get_leg('vL').to_LegCharge()
            B.legs[B.get_leg_index('vR')] = B.get_leg('vR').to_LegCharge()
            mps._B[i] = B#.itranspose(('vL', 'p', 'vR'))
        return mps

    def run(self, init=None):
        self.fidelity = 1
        if self.n_omode > 0:
            print("initialize the mpo-mps calculation")
            self.init_mps(init=init)
            self.n_omode = 0
        nmode = self._U.shape[0]

        for _ in range(nmode):
            u = self._U[_, :]
            v = self._V[_, :]
            # print(self.psi._B)
            self.psi = self.mpomps_step(u, v)
            err = self.psi.compress_svd(self.trunc_params)
            # print(self.psi._B)
            self.fidelity *= 1-err.eps
            self.n_omode += 1
            self.chi_max = np.max(self.psi.chi)
            print( "applied the {}-th mode, the fidelity is {}, the bond dimension is {}".format( self.n_omode, self.fidelity, self.chi_max) )

class MPOMPSU1():

    def __init__(self, u, **kwargs):
        self.cons_N = kwargs.get("cons_N", "N")
        self.cons_S = kwargs.get("cons_S", None)
        self.trunc_params = kwargs.get("trunc_params", dict(chi_max=4) )
        assert u.ndim == 2
        self._U = u
        self.projection_type = kwargs.get("projection_type", "Gutz")
        if self.projection_type == "Gutz":
            self.L = self.Llat = u.shape[1]//2 
        elif self.projection_type == "None":
            self.L = self.Llat = u.shape[1]//2 
        elif self.projection_type == "Ancilla":
            self.L = u.shape[1]//2 
            self.Llat = self.L//3
        self.site = Eletron(self.cons_N, self.cons_S)
        self.init_mps()

    def init_mps(self, init=None):
        L = self.L
        if init is None:
            init = [0] * L
        site = self.site
        self.init_psi = MPS.from_product_state([site]*L, init)
        self.psi = self.init_psi.copy()
        self.n_omode = 0
        return self.psi

    def calc_mpo_u1sz(self, u, s=0):
        chinfo = self.site.leg.chinfo
        pleg = self.site.leg
        if s == 0:
            parity = [1, 1] 
        elif s == 1:
            parity = [1, -1]
        else:
            raise "s should be 0 (up) or 1 (dn)"
        ileg = npc.LegCharge.from_qflat(chinfo, [[0, 0]])
        bleg = npc.LegCharge.from_qflat(chinfo, [parity, [0, 0]])
        fleg = npc.LegCharge.from_qflat(chinfo, [parity])

        leg_frst = [ileg, bleg.conj(), pleg, pleg.conj()]
        leg_bulk = [bleg, bleg.conj(), pleg, pleg.conj()]
        leg_last = [bleg, fleg.conj(), pleg, pleg.conj()]

        mpo = []

        if s == 0:
            assert (np.abs(u[1::2])+np.abs(u[1::2])).sum() < 1e-12
        else:
            assert (np.abs(u[::2])+np.abs(u[::2])).sum() < 1e-12

        L = u.shape[0]//2
        mpo = []

        t0 = npc.zeros( leg_frst, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype )
        if s == 0:
            t0[0, 0, 1, 0] = u[0]
            t0[0, 0, 3, 2] = u[0]
        else:
            t0[0, 0, 2, 0] = u[1]
            t0[0, 0, 3, 1] = -u[1]
        t0[0, 1, 0, 0] = 1;
        t0[0, 1, 1, 1] = -1;
        t0[0, 1, 2, 2] = -1;
        t0[0, 1, 3, 3] = 1;
        mpo.append(t0)

        for i in range(1, L-1):
            ti = npc.zeros( leg_bulk, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype )
            ti[0, 0, 0, 0] = 1;
            ti[0, 0, 1, 1] = 1;
            ti[0, 0, 2, 2] = 1;
            ti[0, 0, 3, 3] = 1;
            if s == 0:
                ti[1, 0, 1, 0] = u[2*i+0]
                ti[1, 0, 3, 2] = u[2*i+0]
            else:
                ti[1, 0, 2, 0] = u[2*i+1]
                ti[1, 0, 3, 1] = -u[2*i+1]
            ti[1, 1, 0, 0] = 1;
            ti[1, 1, 1, 1] = -1;
            ti[1, 1, 2, 2] = -1;
            ti[1, 1, 3, 3] = 1;
            mpo.append(ti)
        i = L-1
        tL = npc.zeros( leg_last, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype )
        tL[0, 0, 0, 0] = 1;
        tL[0, 0, 1, 1] = 1;
        tL[0, 0, 2, 2] = 1;
        tL[0, 0, 3, 3] = 1;
        if s == 0:
            tL[1, 0, 1, 0] = u[2*i+0]
            tL[1, 0, 3, 2] = u[2*i+0]
        else:
            tL[1, 0, 2, 0] = u[2*i+1]
            tL[1, 0, 3, 1] = -u[2*i+1]
        mpo.append(tL)
        return mpo

    def calc_mpo_u1(self, u):
        chinfo = self.site.leg.chinfo
        pleg = self.site.leg
        ileg = npc.LegCharge.from_qflat(chinfo, [[0]])
        bleg = npc.LegCharge.from_qflat(chinfo, [[1], [0]])
        fleg = npc.LegCharge.from_qflat(chinfo, [[1]])
        leg_frst = [ileg, bleg.conj(), pleg, pleg.conj()]
        leg_bulk = [bleg, bleg.conj(), pleg, pleg.conj()]
        leg_last = [bleg, fleg.conj(), pleg, pleg.conj()]

        mpo = []

        L = u.shape[0]//2
        mpo = []

        t0 = npc.zeros( leg_frst, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype )
        t0[0, 0, 1, 0] = u[0]
        t0[0, 0, 3, 2] = u[0]
        t0[0, 0, 2, 0] = u[1]
        t0[0, 0, 3, 1] = -u[1]
        t0[0, 1, 0, 0] = 1;
        t0[0, 1, 1, 1] = -1;
        t0[0, 1, 2, 2] = -1;
        t0[0, 1, 3, 3] = 1;
        mpo.append(t0)

        for i in range(1, L-1):
            ti = npc.zeros( leg_bulk, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype )
            ti[0, 0, 0, 0] = 1;
            ti[0, 0, 1, 1] = 1;
            ti[0, 0, 2, 2] = 1;
            ti[0, 0, 3, 3] = 1;
            ti[1, 0, 1, 0] = u[2*i+0]
            ti[1, 0, 3, 2] = u[2*i+0]
            ti[1, 0, 2, 0] = u[2*i+1]
            ti[1, 0, 3, 1] = -u[2*i+1]
            ti[1, 1, 0, 0] = 1;
            ti[1, 1, 1, 1] = -1;
            ti[1, 1, 2, 2] = -1;
            ti[1, 1, 3, 3] = 1;
            mpo.append(ti)
        i = L-1
        tL = npc.zeros( leg_last, labels=['wL', 'wR', 'p', 'p*'], dtype=u.dtype )
        tL[0, 0, 0, 0] = 1;
        tL[0, 0, 1, 1] = 1;
        tL[0, 0, 2, 2] = 1;
        tL[0, 0, 3, 3] = 1;
        tL[1, 0, 1, 0] = u[2*i+0]
        tL[1, 0, 3, 2] = u[2*i+0]
        tL[1, 0, 2, 0] = u[2*i+1]
        tL[1, 0, 3, 1] = -u[2*i+1]
        mpo.append(tL)
        return mpo        

    def calc_mpo(self, u, v=None):
        if self.cons_N == "N" and self.cons_S == "2*Sz":
            s = 0;
            if abs(u[s::2]).sum() < 1e-12:
                s = 1
            print("s = ", s)
            return self.calc_mpo_u1sz(u, s)
        elif self.cons_N == "N":
            return self.calc_mpo_u1(u)

    def mpomps_step(self, u):
        if isinstance(u, int):
            u = self._U[u, :]
        mpo = self.calc_mpo(u)
        mps = self.psi
        for i in range( self.L ):
            B = npc.tensordot(mps.get_B(i, 'B'), mpo[i], axes=('p', 'p*'))
            B = B.combine_legs([['wL', 'vL'], ['wR', 'vR']], qconj=[+1, -1])
            B.ireplace_labels(['(wL.vL)', '(wR.vR)'], ['vL', 'vR'])
            B.legs[B.get_leg_index('vL')] = B.get_leg('vL').to_LegCharge()
            B.legs[B.get_leg_index('vR')] = B.get_leg('vR').to_LegCharge()
            mps._B[i] = B#.itranspose(('vL', 'p', 'vR'))
        return mps

    def run(self, init=None):
        self.fidelity = 1
        if self.n_omode > 0:
            print("initialize the mpo-mps calculation")
            self.init_mps(init=init)
            self.n_omode = 0
        nmode = self._U.shape[0]

        for _ in range(nmode):
            u = self._U[_, :]
            # print(self.psi._B)
            self.psi = self.mpomps_step(u)
            err = self.psi.compress_svd(self.trunc_params)
            # print(self.psi._B)
            self.fidelity *= 1-err.eps
            self.n_omode += 1
            self.chi_max = np.max(self.psi.chi)
            print( "applied the {}-th mode, the fidelity is {}, the bond dimension is {}".format( self.n_omode, self.fidelity, self.chi_max) )

    # def singlet_projection(self):
    #     chinfo = self.site.leg.chinfo
    #     vleg = self.site.leg
    #     if self.cons_S == '2*Sz':
    #         qtotal = [-2, 0]
    #     else:
    #         qtotal = [-2]
    #     # pleg = 

    #     projector = npc.zeros( [vleg.conj(), vleg.conj()], qtotal=qtotal, labels=['pp1*', 'pp2*'], dtype=self._U.dtype )
    #     # print( projector[1, 2] )
    #     projector[1, 2] =  1/np.sqrt(2)
    #     projector[2, 1] = -1/np.sqrt(2)
    #     self.projector = projector
    #     self.sp_psi = MPS.from_product_state([self.site]*self.Llat, [0]*self.Llat)

    #     for _ in range(self.Llat):
    #         t1 = self.psi._B[_*3+0]
    #         t2 = self.psi._B[_*3+1]
    #         t3 = self.psi._B[_*3+2]
    #         t2.ireplace_label('p','pp1')
    #         t3.ireplace_label('p','pp2')
    #         t23 = npc.tensordot(t2, t3, axes=(['vR'],['vL']))
    #         t23 = npc.tensordot(t23, projector, axes=(['pp1', 'pp2'],['pp1*', 'pp2*']) )
    #         t1 = npc.tensordot(t1, t23, axes=(['vR'],['vL']))
    #         self.sp_psi.set_B(_, t1, form=None)

    #     self.sp_psi.canonical_form()

    # def gutzwiller_projection(self):
    #     chinfo = self.site.leg.chinfo
    #     vleg = self.site.leg
    #     if self.cons_S == '2*Sz':
    #         qtotal = [0, 0]
    #     else:
    #         qtotal = [0]
    #     projector = npc.zeros( [vleg, vleg.conj()], qtotal=[0,0], labels=['p', 'p*'], dtype=self._U.dtype )
    #     projector[1,1] = 1
    #     projector[2,2] = 1

    #     self.gp_psi = MPS.from_product_state([self.site]*self.psi.L, [0]*self.psi.L)
    #     for _ in range(self.psi.L):
    #         t1 = npc.tensordot(self.psi._B[_], projector, axes=(['p'],['p*']))
    #         self.gp_psi.set_B(_, t1, form=None)

    #     self.gp_psi.canonical_form()

def gutzwiller_projection(psi):
    site = psi.sites[0]
    vleg = site.leg
    chinfo = vleg.chinfo
    cons_N, cons_S = site.conserve    
    if cons_S == '2*Sz':
        qtotal = [0, 0]
    else:
        qtotal = [0]

    projector = npc.zeros( [vleg, vleg.conj()], qtotal=qtotal, labels=['p', 'p*'], dtype=psi.dtype )
    projector[1,1] = 1
    projector[2,2] = 1
    L = psi.L 
    gp_psi = MPS.from_product_state([site]*L, [0]*L)
    for _ in range(L):
        t1 = npc.tensordot(psi._B[_], projector, axes=(['p'],['p*']))
        gp_psi.set_B(_, t1, form=None)

    gp_psi.canonical_form()
    return gp_psi

def singlet_projection_(psi):
    site = psi.sites[0]
    vleg = site.leg
    chinfo = vleg.chinfo
    cons_N, cons_S = site.conserve
    qtotal = [-2, 0]
    if cons_N == "N" and cons_S == '2*Sz':
        qtotal = [-2, 0]
    elif cons_N == "N" and cons_S is None:
        qtotal = [-2]
    elif cons_N == 'Z2' and cons_S == '2*Sz':
        qtotal = [0, 0]
    elif cons_N == 'Z2' and cons_S is None:
        qtotal = [0]

    L = psi.L // 3

    projector = npc.zeros( [vleg.conj(), vleg.conj()], qtotal=qtotal, labels=['pp1*', 'pp2*'], dtype=psi.dtype )
    projector[1, 2] =  1/np.sqrt(2)
    projector[2, 1] = -1/np.sqrt(2)
    sp_psi = MPS.from_product_state([site]*L, [0]*L)

    for _ in range(L):
        t1 = psi._B[_*3+0]
        t2 = psi._B[_*3+1]
        t3 = psi._B[_*3+2]
        t2.ireplace_label('p','pp1')
        t3.ireplace_label('p','pp2')
        t23 = npc.tensordot(t2, t3, axes=(['vR'],['vL']))
        t23 = npc.tensordot(t23, projector, axes=(['pp1', 'pp2'],['pp1*', 'pp2*']) )
        t1 = npc.tensordot(t1, t23, axes=(['vR'],['vL']))
        sp_psi.set_B(_, t1, form=None)

    sp_psi.canonical_form()
    return sp_psi

def calc_fermion_swap(legv, legp):
    fswap = npc.zeros((legv, legp, legv.conj(),legp.conj()), labels=('fvR','sp','fvL','sp*'))
    for _v in range(legv.block_number):
        cv = legv.charges[_v]*legv.qconj
        pv = 1 - 2*(cv[0]%2)
        qv = legv.get_qindex_of_charges(cv)
        sv = legv.get_slice(qv)
        # print(pv, cv, qv, sv)
        for _p in range(legp.block_number):
            cp = legp.charges[_p]*legp.qconj
            pp = 1 - 2*(cp[0]%2)
            qp = legp.get_qindex_of_charges(cp)
            sp = legp.get_slice(qp)
            # print(pp, cp, qp, sp)
            val = pp & pv
            for ip in range(sp.start, sp.stop):
                for iv in range(sv.start, sv.stop):
                    fswap[iv,ip,iv,ip] = val
    return fswap

def calc_fermion_swapz2(legv, legp):
    fswap = npc.zeros((legv, legp, legv.conj(),legp.conj()), labels=('fvR','sp','fvL','sp*'))
    for _v in range(legv.block_number):
        cv = legv.charges[_v]*legv.qconj
        pv = 1 - 2*(cv[0]%2)
        qv = legv.get_qindex_of_charges(cv)
        sv = legv.get_slice(qv)
        # print(pv, cv, qv, sv)
        for _p in range(legp.block_number):
            cp = legp.charges[_p]*legp.qconj
            pp = 1 - 2*(cp[0]%2)
            # qp = legp.get_qindex_of_charges(cp)
            sp = legp.get_slice(_p)
            # print(pp, cp, qp, sp)
            val = pp & pv
            for ip in range(sp.start, sp.stop):
                for iv in range(sv.start, sv.stop):
                    fswap[iv,ip,iv,ip] = val
    return fswap

def singlet_projection(fpsi, spsi):
    sites = spsi.sites[0]
    sitef = fpsi.sites[0]
    legPs = spsi.sites[0].leg
    legPf = fpsi.sites[0].leg
    cons_N = sites.cons_N
    cons_S = sites.cons_S

    if cons_N == "N" and cons_S == '2*Sz':
        qtotal = [-2, 0]
    if cons_N == "N" and cons_S == 'parity':
        qtotal = [-2, 0]
    elif cons_N == "N" and cons_S is None:
        qtotal = [-2]
    elif cons_N == 'Z2' and cons_S == 'parity':
        qtotal = [-2, 0]
    elif cons_N == 'Z2' and cons_S is None:
        qtotal = [-2]
    projector = npc.zeros( [legPf.conj(), legPs.conj()], qtotal=qtotal, labels=['pp1*', 'pp2*'] )
    projector[1, 2] =  1/np.sqrt(2)
    projector[2, 1] = -1/np.sqrt(2)

    L = spsi.L

    sp_psi = MPS.from_product_state([sitef]*L, [0]*L)

    for _ in range(L):
        legp = spsi._B[_].legs[1]
        legv = fpsi._B[2*_+1].legs[2]
        if cons_N == 'N':
            fswap = calc_fermion_swap(legv, legp)
        else:
            fswap = calc_fermion_swapz2(legv, legp)
        t0 = fpsi._B[2*_+0]
        t1 = fpsi._B[2*_+1]
        t2 = spsi._B[_]
        t = npc.tensordot(t1, fswap, axes=(['vR'],['fvL']))
        t.ireplace_labels(['p', 'vL'], ['fp','fvL'])
        t = npc.tensordot(t, spsi._B[_], axes=(['sp*'],['p']))
        t.ireplace_labels(['vR', 'vL'], ['svR','svL'])
        t = npc.tensordot(t, projector, axes=(['fp', 'sp'],['pp1*', 'pp2*']) )
        t = npc.tensordot(t0, t, axes=(['vR'],['fvL']) )
        t = t.combine_legs(['vL','svL'],qconj=+1)
        t = t.combine_legs(['fvR','svR'],qconj=-1)
        t.ireplace_labels(['(vL.svL)', '(fvR.svR)'], ['vL','vR'])
        sp_psi.set_B(_, t, form=None)   
    sp_psi.canonical_form()
    return sp_psi

if __name__ == "__main__":
    from PartonTriangle import *
    from PartonHoneycomb import *
    lx = 2
    ly = 2
    t = 1
    bcy = 1.
    P = np.inf
    chi = 128

    params_spin = dict(lx=lx, ly=ly, bcy=bcy, bcx=0, Jx=1, Jy=1, Jz=1)
    model = HoneycombKitaev(params_spin)
    fu, fv = model.calc_wannier_state()
    chi=128
    params_mpompsz2 = dict(cons_N="Z2", cons_S=None, trunc_params=dict(chi_max=chi))
    eng_spin = MPOMPSZ2(fu, fv, **params_mpompsz2)
    eng_spin.run()
    gpsi_spin = gutzwiller_projection(eng_spin.psi)
    ops = ["Id"]*gpsi_spin.L
    ops[1]=ops[6]='X';
    ops[3]=ops[4]='Z';
    ops[2]=ops[5]='Y';
    gpsi_spin.expectation_value_multi_sites(ops, 0)

    params_flux = dict(lx=lx, ly=ly, bcy=bcy, bcx=0, tx=1, ty=1, tz=1, tc=t, tf=0, uc=0,uf=0, P=np.inf)
    model_flux = FluxHoneycomb(params_flux)
    fu = model_flux.calc_u_real()
    params_mpompsu1 = dict(cons_N="N", cons_S=None, trunc_params=dict(chi_max=chi))
    eng_flux = MPOMPSU1(fu, **params_mpompsu1)
    eng_flux.run()
    psi_flux = eng_flux.psi
    sp_psi = singlet_projection(psi_flux, gpsi_spin)

    # F = 0.15
    # ansatz = 'CSL'
    # params_spin = dict(lx=lx, ly=ly, bcy=bcy, bcx=0, F=F)
    # params_flux = dict(lx=lx, ly=ly, bcy=bcy, bcx=0, P=P)
    # model_params = dict(params_spin=params_spin, params_flux=params_flux, spin_ansatz=ansatz)    
    # print( model_params )
    # cons_S = "2*Sz"

    # params_mpomps = dict(cons_N="N", cons_S=cons_S, trunc_params=dict(chi_max=chi), projection_type='Ancilla')
    # model = PartonTriangle( model_params )
    # model.calc_wannier_state()
    # uF = model.wannier_all 
    # eng = MPOMPSU1(uF, **params_mpomps)
    # eng.run()
    # sp_psi1 = singlet_projection_(eng.psi)


    # params_mpomps = dict(cons_N="N", cons_S=cons_S, trunc_params=dict(chi_max=chi), projection_type='Gutz')
    # uF = model.wannier_spin
    # eng2 = MPOMPSU1(uF, **params_mpomps)
    # eng2.run()
    # gp_psi1 = gutzwiller_projection(eng2.psi)

    # params_mpomps = dict(cons_N="N", cons_S=cons_S, trunc_params=dict(chi_max=chi), projection_type='None')
    # uF = model.wannier_flux
    # eng3 = MPOMPSU1(uF, **params_mpomps)
    # eng3.run()
    # sp_psi2 = singlet_projection(eng3.psi, eng2.psi)

    # print( gp_psi1.overlap(sp_psi2) )
    # print( sp_psi1.overlap(sp_psi2) )
    # print( gp_psi1.overlap(sp_psi1) )


