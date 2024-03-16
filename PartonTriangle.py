import numpy as np
import scipy as sp
import sys, os
import pickle
from matplotlib import pyplot as plt

class TriangleLatt():

    def __init__(self, model_params=dict()):
        self.t = model_params.get("t", 1.)
        self.U = model_params.get('U', 10.)
        self.lx = model_params.get("lx", 2)
        self.ly = model_params.get("ly", 2)
        self.Nlat = self.lx * self.ly
        self.bcx = model_params.get('bcx', 0.)
        self.bcy = model_params.get('bcy', 1.)

        self.init_lat()

    def _xy2id(self, y, x, Y=None, X=None):
        if Y is None:
            Y = self.ly
        if X is None:
            X = self.lx
        return (y % Y) + (x % X)*Y

    def init_lat(self):
        lx, ly = self.lx, self.ly
        self.r1 = np.array([1, 0])
        self.r2 = np.array([1/2, np.sqrt(3)/2])

        self.xy2id = dict()
        self.id2xy = dict()
        self.xy2nn = dict()
        self.id2nn = dict()
        # self.bcxid = set()
        # self.bcyid = set()
        self.bcxnn = set()
        self.bcynn = set()
        for _x in range(lx):
            for _y in range(ly):
                xy = (_x, _y)
                idxy = self._xy2id(_y, _x)
                self.xy2id[xy] = idxy
                self.id2xy[idxy] = xy
                nnpy = self._xy2id(_y+1, _x) 
                nnpx = self._xy2id(_y, _x+1)
                nnmy = self._xy2id(_y-1, _x) 
                nnmx = self._xy2id(_y, _x-1)
                nnmxpy = self._xy2id(_y+1, _x-1)
                nnpxmy = self._xy2id(_y-1, _x+1)
                self.id2nn[idxy] = [nnpy, nnpx, nnpxmy, nnmy, nnmx, nnmxpy] # note the order of nn
                if _x == lx-1:
                    self.bcxnn.add(  tuple(np.sort([idxy, nnpx]).tolist()) )
                    self.bcxnn.add(  tuple(np.sort([idxy, nnpxmy]).tolist()) )
                if _y == 0:
                    self.bcynn.add(  tuple(np.sort([idxy, nnmy]).tolist()) )
                    self.bcynn.add(  tuple(np.sort([idxy, nnpxmy]).tolist()) )

    def plot_lat_basic(self, ax=None):
        if ax is None:
            ax = plt.subplot(111)
        r1, r2 = self.r1, self.r2
        for id0 in self.id2xy.keys():
            xy = self.id2xy[id0]
            r0 = xy[0]*r1+xy[1]*r2
            ax.text(r0[0], r0[1], str(id0), c='k')
            for __, nn in enumerate( self.id2nn[id0][:3]):
                xynn = self.id2xy[nn]
                rnn = xynn[0]*r1+xynn[1]*r2
                bond = tuple(np.sort([id0, nn]).tolist())
                if bond in self.bcxnn and bond not in self.bcynn:
                    if __ == 1:
                        ax.plot([r0[0], r0[0]+1], [r0[1], rnn[1]], '--', c='k')
                    if __ == 2:
                        ax.plot([r0[0], r0[0]+0.5], [r0[1], rnn[1]], '--', c='k')
                elif bond in self.bcynn:
                    if __ == 0:
                        ax.plot([r0[0], r0[0]+0.5], [r0[1], r0[1]+np.sqrt(3)/2], '--', c='k')
                    if __ == 2:
                        ax.plot([rnn[0]+0.5,rnn[0]+1], [rnn[1]+np.sqrt(3)/2,rnn[1]], '--', c='k')
                else:
                    ax.plot([r0[0], rnn[0]], [r0[1], rnn[1]], '-', c='k')

    def calc_real_ham(self):
        tc = 1
        N, lx, ly = self.Nlat, self.lx, self.ly
        bcx, bcy = self.bcx, self.bcy
        # print("bcx: ", bcx)
        # print("bcy: ", bcy)

        ham = np.zeros((N, N), float)

        for _x in range(lx):
            for _y in range(ly):
                id0    = self._xy2id(_y,   _x)
                idpx   = self._xy2id(_y,   _x+1)
                idpy   = self._xy2id(_y+1, _x)
                idmxpy = self._xy2id(_y+1, _x-1)

                if _y == ly-1 and _x == 0:
                    ham[id0+0, idmxpy+0] = -tc*bcy*bcx
                    # print(id0+0, idmxpy+0, tc*bcy*bcx, bcx, bcy)
                elif _y == ly-1 and _x !=0:
                    ham[id0+0, idmxpy+0] = -tc*bcy                
                elif _x == 0 and _y != ly-1:
                    ham[id0+0, idmxpy+0] = -tc*bcx
                else:
                    ham[id0+0, idmxpy+0] = -tc          

                if _y == ly-1:
                    ham[id0+0, idpy+0] = -tc*bcy
                else:
                    ham[id0+0, idpy+0] = -tc
                
                if _x == lx-1:
                    ham[id0+0, idpx+0] = -tc*bcx
                else:
                    ham[id0+0, idpx+0] = -tc

        ham += ham.T.conj()
        self.ham_real = ham
        self.eng_real, self.state_real = np.linalg.eigh(ham)
        return self.eng_real, self.state_real

    def Wannier_U1(self, g1, N=1):
        norbital, n = g1.shape
        position = np.diag( np.power( list(range(1,n+1)), N) )
        position12 = g1.conj() @ position @ g1.T 
        position12 = (position12 + position12.T.conj())/2.
        D, U = np.linalg.eigh(position12)
        index = np.argsort(D)
        # print(D)
        U = U[:,index]
        g3 = U.T @ g1
        index1 = np.zeros(norbital, dtype=int)
        if norbital%2 == 0:
            index1[0:(norbital):2] = np.ceil( range(0, norbital//2) ).astype(int)
            index1[1:(norbital):2] = np.ceil( range(norbital-1, norbital//2-1, -1) ).astype(int)
        else:
            index1[0:(norbital):2] = np.ceil( range(0, norbital//2+1) ).astype(int)
            index1[1:(norbital):2] = np.ceil( range(norbital-1, norbital//2, -1) ).astype(int)
        g3 = g3[index1,:]
        return g3

    def calc_wannier_state(self):
        _, flux_vec = self.calc_real_ham()
        wannier_flux = flux_vec[:, :flux_vec.shape[1]//2]
        wannier_flux = wannier_flux.T
        wannier_flux = self.Wannier_U1(wannier_flux)
        self.wannier_flux = wannier_flux

        N = int(self.Nlat*2); Nmode = int(self.Nlat)

        self.wannier_all = np.zeros((Nmode, N), float)
        print(self.wannier_flux.shape, self.wannier_all.shape)

        for _ in range(wannier_flux.shape[0]):
            self.wannier_all[_*2+0, 0::2] = wannier_flux[_, :]
            self.wannier_all[_*2+1, 1::2] = wannier_flux[_, :]

class NeelTriangle(TriangleLatt):

    def __init__(self, model_params=dict()):
        super(NeelTriangle, self).__init__(model_params)
        self.M = model_params.get("M", 3.0)
        self.sublat = { _:[] for _ in range(3) }        
        for _x in range(self.lx):
            for _y in range(self.ly):
                id0 = self.xy2id[(_x, _y)]
                flag = (_y % 3 + (_x % 3)*2 ) %3
                self.sublat[flag].append( id0 )

    def plot_lat(self, ax=None):
        if ax is None:
            ax = plt.subplot(111)
        self.plot_lat_basic(ax)
        for _ in range(3):
            c = ['r','b','g'][_]
            A = ['A','C','B'][_]
            for idd in self.sublat[_]:
                xy = self.id2xy[idd]
                r0 = xy[0]*self.r1+xy[1]*self.r2
                ax.plot([r0[0]], [r0[1]],'o', c=c)
        plt.axis('equal')
        plt.show()

    def calc_real_ham(self):
        t, M = self.t, self.M
        bcx, bcy = self.bcx, self.bcy
        N, lx, ly = self.Nlat, self.lx, self.ly

        ham = np.zeros((N*2, N*2), complex)
        for id0 in self.id2nn:
            for id1 in self.id2nn[id0][:3]:
                bond = tuple(np.sort([id0, id1]).tolist())
                if bond in self.bcxnn and bond in self.bcynn:
                    ham[id0*2+0, id1*2+0] += -t*bcx*bcy
                    ham[id0*2+1, id1*2+1] += -t*bcx*bcy
                elif bond in self.bcxnn:
                    ham[id0*2+0, id1*2+0] += -t*bcx
                    ham[id0*2+1, id1*2+1] += -t*bcx
                elif bond in self.bcynn:
                    ham[id0*2+0, id1*2+0] += -t*bcy
                    ham[id0*2+1, id1*2+1] += -t*bcy
                else:
                    ham[id0*2+0, id1*2+0] += -t
                    ham[id0*2+1, id1*2+1] += -t
            if id0 in self.sublat[0]:
                ham[id0*2+0, id0*2+1] += -M*(-np.sqrt(3)/2+1j/2)
            elif id0 in self.sublat[1]:
                ham[id0*2+0, id0*2+1] += -M*(-1j)
            elif id0  in self.sublat[2]:
                ham[id0*2+0, id0*2+1] += -M*(+np.sqrt(3)/2+1j/2)
        ham += ham.T.conj()
        self.ham_real = ham
        self.eng_real, self.state_real = np.linalg.eigh(ham)
        return self.eng_real, self.state_real

    def calc_k_ham(self, k1, k2):
        vk1, vk2 = k1*np.pi*2/(self.lx/np.sqrt(3)), k2*np.pi*2/(self.ly/np.sqrt(3))
        t, M = self.t, self.M
        ham = np.zeros((6,6),complex)
        sx = np.array([[0,1],[1,0]]); sy = np.array([[0,-1j],[1j,0]]); s0 = np.diag([1,1])
        ham[0:2, 0:2] = -M * (-np.sqrt(3)*sx-sy)/2
        ham[2:4, 2:4] = -M * (+np.sqrt(3)*sx-sy)/2
        ham[4:6, 4:6] = -M * (+sy)
        dAB = -t*( 1+np.exp(-1j*vk1)+np.exp(-1j*(vk1-vk2)) )
        dBC = -t*( 1+np.exp(-1j*vk2)+np.exp(-1j*(vk2-vk1)) )
        dCA = -t*( 1+np.exp(+1j*vk2)+np.exp(+1j*vk1) )
        ham[0:2, 2:4] = dAB*s0
        ham[2:4, 0:2] = dAB.conj()*s0
        ham[0:2, 4:6] = dCA.conj()*s0
        ham[4:6, 0:2] = dCA*s0
        ham[2:4, 4:6] = dBC*s0
        ham[4:6, 2:4] = dBC.conj()*s0
        eng, state = np.linalg.eigh(ham)
        if not hasattr(self, 'eng_k'):
            self.eng_k = dict()
            self.state_k = dict()
        self.eng_k[(k1, k2)] = eng 
        self.state_k[(k1, k2)] = state
        return eng, state

    def plot_path_Gamma_GKMG(self):
        x_point = 170
        GK = 1/np.sqrt(3)*np.linspace(0, self.lx, int(x_point/17*20))/3
        KM = 1/np.sqrt(3)*np.linspace(self.lx, 0, int(x_point/17*10))/3
        MG = 1/np.sqrt(3)*np.linspace(self.lx, 0, int(x_point))/3
        engs = []
        for _ in range(len(GK)):
            e, v = self.calc_k_ham(GK[_], -GK[_])
            engs.append(e)
        for _ in range(len(KM)):
            e, v = self.calc_k_ham(GK[-1], -KM[_])
            engs.append(e)
        for _ in range(len(MG)):
            e, v = self.calc_k_ham(MG[_], 0)
            engs.append(e)
        for _ in range(len(GK)):
            e, v = self.calc_k_ham(GK[_], GK[_])
            engs.append(e)
        plt.plot(np.array(engs))
        plt.xticks([0, len(GK),len(GK)+len(KM),len(GK)+len(KM)+len(MG),len(GK)*2+len(KM)+len(MG)],[r'$\Gamma$',r'$K$',r'$M$',r'$\Gamma$',r'$K^{\prime}$'])
        plt.show()

class CSLTriangle(TriangleLatt):

    def __init__(self, model_params=dict()):
        super(CSLTriangle, self).__init__(model_params)
        self.F = model_params.get("F", 0.05)
        self.sublat = { _:[] for _ in range(2) }        
        for _x in range(self.lx):
            for _y in range(self.ly):
                id0 = self.xy2id[(_x, _y)]
                flag = (_x % 2) 
                self.sublat[flag].append( id0 )

    def plot_lat(self, ax=None):
        if ax is None:
            ax = plt.subplot(111)
        self.plot_lat_basic(ax)
        for _ in range(2):
            c = ['r','b'][_]
            A = ['A','B'][_]
            for idd in self.sublat[_]:
                xy = self.id2xy[idd]
                r0 = xy[0]*self.r1+xy[1]*self.r2
                ax.plot([r0[0]], [r0[1]],'o', c=c)
        plt.axis('equal')
        plt.show()

    def calc_real_ham(self):
        t = self.t
        bcx, bcy = self.bcx, self.bcy
        N, lx, ly = self.Nlat, self.lx, self.ly
        Mx = self.lx//2
        F = self.F*2*np.pi
        pi = np.pi

        ham = np.zeros((N, N), complex)

        for _x in range(Mx):
            for _y in range(ly):
                id0    = 2*self._xy2id(_y,   _x,   X=Mx)
                idpx   = 2*self._xy2id(_y,   _x+1, X=Mx)
                idpy   = 2*self._xy2id(_y+1, _x,   X=Mx)
                idmxpy = 2*self._xy2id(_y+1, _x-1, X=Mx)

                ham[id0+0, id0+1] = np.exp(1j*(F+pi))

                if _y == ly-1 and _x == 0:
                    ham[id0+0, idmxpy+1] = np.exp(1j*(F+pi))*bcy*bcx
                elif _y == ly-1:
                    ham[id0+0, idmxpy+1] = np.exp(1j*(F+pi))*bcy                
                else:
                    ham[id0+0, idmxpy+1] = np.exp(1j*(F+pi))

                if _y == ly-1:
                    ham[id0+0, idpy+0]   = np.exp(-1j*(F+pi))*bcy
                else:
                    ham[id0+0, idpy+0]   = np.exp(-1j*(F+pi))

                if _y == ly-1:
                    ham[id0+1, idpy+0]   = np.exp(1j*F)*bcy
                else:
                    ham[id0+1, idpy+0]   = np.exp(1j*F)
                
                if _x == Mx-1:
                    ham[id0+1, idpx+0]   = np.exp(1j*(F+pi))*bcx
                else:
                    ham[id0+1, idpx+0]   = np.exp(1j*(F+pi))*bcx

                if _y == ly-1:
                    ham[id0+1, idpy+1]   = np.exp(-1j*F)*bcy
                else:
                    ham[id0+1, idpy+1]   = np.exp(-1j*F)

        ham += ham.T.conj()
        self.ham_real = ham
        self.eng_real, self.state_real = np.linalg.eigh(ham)
        return self.eng_real, self.state_real

    def calc_k_ham(self, k1, k2):
        vk1, vk2 = k1*np.pi*2/(self.lx/2), k2*np.pi*2/self.ly
        F = self.F*2*np.pi;    
        ham = np.zeros((2,2),complex)
        ham[0, 0] = -2*np.cos(F-vk2)
        ham[1, 1] = +2*np.cos(F-vk2)
        ham[0, 1] = -np.exp(1j*F)*( 1+np.exp(-1j*vk1+1j*vk2) ) + np.exp(-1j*F)*( np.exp(-1j*vk2)-np.exp(-1j*vk1) )
        ham[1, 0] = ham[0, 1].conj()
        eng, state = np.linalg.eigh(ham)
        if not hasattr(self, 'eng_k'):
            self.eng_k = dict()
            self.state_k = dict()
        self.eng_k[(k1, k2)] = eng 
        self.state_k[(k1, k2)] = state
        return eng, state

    def plot_path_Gamma_GKMG(self):
        x_point = 170
        GK = 1/2*np.linspace(0, self.lx, int(x_point/17*10))/2
        KM = np.linspace(0, self.lx,int(x_point))/2
        MG = np.linspace(self.lx, 0, int(x_point/17*20))/2
        engs = []
        for _ in range(len(GK)):
            e, v = self.calc_k_ham(0, GK[_])
            engs.append(e)
        for _ in range(len(KM)):
            e, v = self.calc_k_ham(KM[_], GK[-1])
            engs.append(e)
        for _ in range(len(MG)):
            e, v = self.calc_k_ham(MG[_], MG[_]/2)
            engs.append(e)
        plt.plot(np.array(engs))
        plt.xticks([0, len(GK),len(GK)+len(KM),len(GK)+len(KM)+len(MG)],[r'$\Gamma$',r'$M$',r'$K$',r'$\Gamma$'])
        plt.show()

    def calc_wannier_state(self):
        _, spin_vec = self.calc_real_ham()
        # half filling
        spin_vec = spin_vec[:, :spin_vec.shape[1]//2]
        spin_vec =spin_vec.T
        wannier_spin = self.Wannier_U1(spin_vec)

        Nspin = wannier_spin.shape[0]

        N = int(self.Nlat*2); Nmode = int(Nspin*2)
        print("the symmetry of the wavefunction is SU(2)")        
        print("the order of fermions are s_up, s_dn.")

        self.wannier_all = np.zeros((Nmode, N), complex)

        for _ in range(Nspin):
            self.wannier_all[_*2+0, 0::2] = wannier_spin[_, :]
            self.wannier_all[_*2+1, 1::2] = wannier_spin[_, :]

class FluxTriangle(TriangleLatt):

    def __init__(self, model_params=dict()):
        super(FluxTriangle, self).__init__(model_params)
        self.Pcf = model_params.get("P", np.inf)
        self.tf = model_params.get("tf", 0.)
        self.tc = self.t
        self.uf = model_params.get('uf', 0.)
        self.uc = model_params.get('uc', 0.)

    def calc_real_ham(self):
        tc, tf, uc, uf, Pcf = self.tc, self.tf, self.uc, self.uf, self.Pcf
        N, lx, ly = self.Nlat, self.lx, self.ly
        bcx, bcy = self.bcx, self.bcy

        ham = np.zeros((N*2, N*2), float)

        for _x in range(lx):
            for _y in range(ly):
                id0    = 2*self._xy2id(_y,   _x)
                idpx   = 2*self._xy2id(_y,   _x+1)
                idpy   = 2*self._xy2id(_y+1, _x)
                idmxpy = 2*self._xy2id(_y+1, _x-1)

                ham[id0+0, id0+0] += uc/2
                ham[id0+1, id0+1] += uf/2
                ham[id0+0, id0+1] += Pcf

                if _y == ly-1 and _x == 0:
                    ham[id0+0, idmxpy+0] = -tc*bcy*bcx
                    ham[id0+1, idmxpy+1] = +tf*bcy*bcx
                elif _y == ly-1 and _x !=0:
                    ham[id0+0, idmxpy+0] = -tc*bcy                
                    ham[id0+1, idmxpy+1] = +tf*bcy                
                elif _x == 0 and _y != ly-1:
                    ham[id0+0, idmxpy+0] = -tc*bcx                
                    ham[id0+1, idmxpy+1] = +tf*bcx                    
                else:
                    ham[id0+0, idmxpy+0] = -tc          
                    ham[id0+1, idmxpy+1] = +tf                

                if _y == ly-1:
                    ham[id0+0, idpy+0] = -tc*bcy
                    ham[id0+1, idpy+1] = +tf*bcy
                else:
                    ham[id0+0, idpy+0] = -tc
                    ham[id0+1, idpy+1] = +tf
                
                if _x == lx-1:
                    ham[id0+0, idpx+0] = -tc*bcx
                    ham[id0+1, idpx+1] = +tf*bcx
                else:
                    ham[id0+0, idpx+0] = -tc
                    ham[id0+1, idpx+1] = +tf


        ham += ham.T.conj()
        self.ham_real = ham
        self.eng_real, self.state_real = np.linalg.eigh(ham)
        return self.eng_real, self.state_real

    def calc_u_real(self, Pcf=None):
        if Pcf is None:
            Pcf = self.Pcf
        if Pcf == np.inf:
            N, lx, ly = self.Nlat, self.lx, self.ly
            self.state_real = np.zeros((N*2, N*2), float)
            rs2 = 1/np.sqrt(2)
            for _ in range(N):
                id0 = _ * 2
                self.state_real[id0+0, _] = +rs2
                self.state_real[id0+1, _] = -rs2
                self.state_real[id0+0, _+N] = +rs2
                self.state_real[id0+1, _+N] = +rs2
            return self.state_real.T
        elif 1e-12 < abs(self.Pcf) < self.tc:
            print("Warning! ")
        elif abs(self.Pcf) < 1e-12:
            N, lx, ly = self.Nlat, self.lx, self.ly
            self.state_real = np.zeros((N*2, N*2), float)
            rs2 = 1/np.sqrt(2)
            m = TriangleLatt(dict(lx=self.lx, ly=self.ly, bcx=self.bcx, bcy=self.bcy))
            m.calc_wannier_state()
            uf = m.wannier_flux
            self.state_real = np.zeros((N*2, N*2), float)
            for _ in range(N//2):
                self.state_real[_*4+1, _] = +rs2
                self.state_real[_*4+3, _] = -rs2
            for _ in range(uf.shape[0]):
                self.state_real[0::2, N//2+_] = uf[_, :]
            return self.state_real.T
        else:
            self.calc_real_ham()
            return self.state_real.T

class AncillaTriangle():

    def __init__(self, model_params=dict()):
        self.model_params = model_params
        self.params_flux = self.model_params.get('params_flux', dict())
        self.params_spin = self.model_params.get('params_spin', dict())
        self.spin_ansatz = self.model_params.get('spin_ansatz', 'CSL')
        if self.spin_ansatz == 'CSL':
            self.spin_model = CSLTriangle(self.params_spin)
            self.spin_symmetry = 'Sz'
            print('the spinon part ansatz is a CSL')
        elif self.spin_ansatz == '120':
            self.spin_model = NeelTriangle(self.params_spin)
            self.spin_symmetry = 'None'
            print('the spinon part ansatz is a 120 Neel order')
        else:
            print('the ansatz should be CSL or 120')
            raise None

        self.flux_model = FluxTriangle(self.params_flux)
        assert self.spin_model.lx == self.flux_model.lx
        assert self.spin_model.ly == self.flux_model.ly
        assert abs(self.spin_model.bcx) == abs(self.flux_model.bcx)
        assert abs(self.spin_model.bcy) == abs(self.flux_model.bcy)
        self.Nlat = self.spin_model.Nlat
        self.lx = self.spin_model.lx
        self.ly = self.spin_model.ly

    def Wannier_U1(self, g1, N=1):
        norbital, n = g1.shape
        position = np.diag( np.power( list(range(1,n+1)), N) )
        position12 = g1.conj() @ position @ g1.T 
        position12 = (position12 + position12.T.conj())/2.
        D, U = np.linalg.eigh(position12)
        index = np.argsort(D)
        # print(D)
        U = U[:,index]
        g3 = U.T @ g1
        index1 = np.zeros(norbital, dtype=int)
        if norbital%2 == 0:
            index1[0:(norbital):2] = np.ceil( range(0, norbital//2) ).astype(int)
            index1[1:(norbital):2] = np.ceil( range(norbital-1, norbital//2-1, -1) ).astype(int)
        else:
            index1[0:(norbital):2] = np.ceil( range(0, norbital//2+1) ).astype(int)
            index1[1:(norbital):2] = np.ceil( range(norbital-1, norbital//2, -1) ).astype(int)
        g3 = g3[index1,:]
        return g3

    def calc_wannier_state(self):
        flux_vec = self.flux_model.calc_u_real()
        flux_vec = flux_vec[:flux_vec.shape[1]//2, :]
        if self.flux_model.Pcf == np.inf:
            wannier_flux = flux_vec
        elif self.flux_model.Pcf == 0:
            wannier_flux = flux_vec
        else:
            wannier_flux = self.Wannier_U1(flux_vec)
        _, spin_vec = self.spin_model.calc_real_ham()
        spin_vec = spin_vec[:, :spin_vec.shape[1]//2]
        wannier_spin = self.Wannier_U1(spin_vec)
        # self.wannier_flux, self.wannier_spin = wannier_flux, wannier_spin   

        N = int(self.Nlat*2*3); Nmode = int(self.Nlat*3)
        print("the symmetry of the wavefunction is ", self.spin_symmetry)        
        print("the order of fermions are c_up, c_dn, f_up, f_dn, s_up, s_dn, where")
        print('c : electron,    f : acilla charge,     s : spinon')

        Nflux = wannier_flux.shape[0]
        Nspin = wannier_spin.shape[0]

        self.wannier_spin = np.zeros((Nspin*2, self.Nlat*2), complex)
        for _ in range(Nspin):
            self.wannier_spin[_*2+0, 0::2] = wannier_spin[_, :]
            self.wannier_spin[_*2+1, 1::2] = wannier_spin[_, :]

        self.wannier_flux = np.zeros((Nflux*2, self.Nlat*4), complex)
        for _ in range(Nflux):
            self.wannier_flux[_*2+0, 0::4] = wannier_flux[_, 0::2]
            self.wannier_flux[_*2+0, 2::4] = wannier_flux[_, 1::2]
            self.wannier_flux[_*2+1, 1::4] = wannier_flux[_, 0::2]
            self.wannier_flux[_*2+1, 3::4] = wannier_flux[_, 1::2]

        self.wannier_all = np.zeros((Nmode, N), complex)
        print(self.wannier_flux.shape,self.wannier_spin.shape, self.wannier_all.shape)

        for _ in range( Nflux ):
            self.wannier_all[_*2+0, 0::6] = wannier_flux[_, 0::2]
            self.wannier_all[_*2+0, 2::6] = wannier_flux[_, 1::2]
            self.wannier_all[_*2+1, 1::6] = wannier_flux[_, 0::2]
            self.wannier_all[_*2+1, 3::6] = wannier_flux[_, 1::2]

        if Nspin == Nflux:
            print('the spinon part does not have SU(2) symmetry')
            for _ in range(Nspin):
                self.wannier_all[2*Nflux+_, 4::6] = wannier_spin[_, 0::2]
                self.wannier_all[2*Nflux+_, 5::6] = wannier_spin[_, 1::2]
        else:
            print('the spinon part has SU(2) symmetry')
            for _ in range(Nspin):
                self.wannier_all[2*Nflux+_*2+0, 4::6] = wannier_spin[_, :]
                self.wannier_all[2*Nflux+_*2+1, 5::6] = wannier_spin[_, :]



# lx = 4; ly = 4
# params_spin = dict(lx=lx, ly=ly, bcy=1, bcx=0, F=0)
# params_flux = dict(lx=lx, ly=ly, bcy=1, bcx=0, P=0.)
# model_params = dict(params_spin=params_spin, params_flux=params_flux, spin_ansatz='CSL')    
# model = PartonTriangle(model_params)
# model.calc_wannier_state()
# todel = TriangleLatt(params_flux)
# todel.calc_wannier_state()
# print(model.eng_real[:6])
# print(model.eng_real[:6].sum()*2)