import numpy as np
import scipy as sp
import sys, os
import pickle
from matplotlib import pyplot as plt

class SquareLatt():

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
        self.r2 = np.array([0, 1])

        self.xy2id = dict()
        self.id2xy = dict()
        for _x in range(lx):
            for _y in range(ly):
                xy = (_x, _y)
                idxy = self._xy2id(_y, _x)
                self.xy2id[xy] = idxy
                self.id2xy[idxy] = xy

    def calc_real_ham(self):
        tc = 1
        N, lx, ly = self.Nlat, self.lx, self.ly
        bcx, bcy = self.bcx, self.bcy

        ham = np.zeros((N, N), float)

        for _x in range(lx):
            for _y in range(ly):
                id0    = self._xy2id(_y,   _x)
                idpx   = self._xy2id(_y,   _x+1)
                idpy   = self._xy2id(_y+1, _x)
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

    def Wannier_Z2(self, g1, g2, N=1):
        norbital, n = g1.shape        
        position = np.power( list(range(1,n+1)), N)
        position = np.diag(position) 
        position12 = g1.conj() @ position @ g1.T + g2.conj() @ position @ g2.T
        position12 = (position12 + position12.T.conj())/2.
        D, U = np.linalg.eigh(position12)
        index = np.argsort(D)
        print("Wannier_Z2", D)
        U = U[:,index]
        g3 = U.T @ g1
        g4 = U.T @ g2
        index1 = np.zeros(norbital, dtype=int)
        if norbital%2 == 0:
            index1[0:(norbital):2] = np.ceil( range(0, norbital//2) ).astype(int)
            index1[1:(norbital):2] = np.ceil( range(norbital-1, norbital//2-1, -1) ).astype(int)
        else:
            index1[0:(norbital):2] = np.ceil( range(0, norbital//2+1) ).astype(int)
            index1[1:(norbital):2] = np.ceil( range(norbital-1, norbital//2, -1) ).astype(int)
        g3 = g3[index1,:]
        g4 = g4[index1,:]
        return g3, g4

    def calc_wannier_state(self):
        _, flux_vec = self.calc_real_ham()
        wannier_flux = flux_vec[:, :flux_vec.shape[1]//2]
        wannier_flux = self.Wannier_U1(wannier_flux.T)
        self.wannier_flux = wannier_flux

        N = int(self.Nlat*2); Nmode = int(self.Nlat)

        self.wannier_all = np.zeros((Nmode, N), float)
        print(self.wannier_flux.shape, self.wannier_all.shape)

        for _ in range(wannier_flux.shape[0]):
            self.wannier_all[_*2+0, 0::2] = wannier_flux[_, :]
            self.wannier_all[_*2+1, 1::2] = wannier_flux[_, :]

class Z2QSLSquare(SquareLatt):

    def __init__(self, model_params=dict()):
        super(Z2QSLSquare, self).__init__(model_params)
        self.dxy = model_params.get('dxy', 0.1)
        self.dxxmyy = model_params.get("dxxmyy", 0.1)

    def calc_real_ham_SU2(self):
        """
        note it is a SU(2)-symmetric s-wave BdG model
        """
        t = self.t
        dxy = self.dxy
        dxxmyy = self.dxxmyy
        print("t=", t, ",  dxy=", dxy, ",  dxxmyy=", dxxmyy)

        bcx, bcy = self.bcx, self.bcy
        N, lx, ly = self.Nlat, self.lx, self.ly

        tmat = np.zeros((N, N), float)
        dmat = np.zeros((N, N), float)
        Ap = [];    Bp = [];    Cp = [];    Dp = []
        Am = [];    Bm = [];    Cm = [];    Dm = []

        for _x in range(lx):
            for _y in range(ly):
                id0      = self._xy2id(_y,   _x)
                idpx     = self._xy2id(_y,   _x+1)
                idpy     = self._xy2id(_y+1, _x)
                idp2xp2y = self._xy2id(_y+2, _x+2)
                idm2xp2y = self._xy2id(_y+2, _x-2)
                if _x == lx - 1:
                    tmat[id0, idpx] += t*bcx 
                    dmat[id0, idpx] += dxxmyy*bcx 
                else:
                    tmat[id0, idpx] += t          # + f^\dag_{id0, up} f_{id0+x, up}
                    dmat[id0, idpx] += dxxmyy     # + f^\dag_{id0, up} f^\dag_{id0+x, down}

                if _y == ly - 1:
                    tmat[id0, idpy] += t*bcy 
                    dmat[id0, idpy] -= dxxmyy*bcy 
                else:
                    tmat[id0, idpy] += t          # + f^\dag_{id0, up} f_{id0+y, up}
                    dmat[id0, idpy] -= dxxmyy     # - f^\dag_{id0, up} f^\dag_{id0+y, down}

                if (_y < ly - 2) and (_x < lx - 2):
                    Ap.append((id0,idp2xp2y))
                    dmat[id0, idp2xp2y] += dxy
                elif _x < lx - 2:
                    Bp.append((id0,idp2xp2y))
                    dmat[id0, idp2xp2y] += dxy*bcy
                elif _y < ly - 2:
                    Cp.append((id0,idp2xp2y))
                    dmat[id0, idp2xp2y] += dxy*bcx
                else:
                    Dp.append((id0,idp2xp2y))
                    dmat[id0, idp2xp2y] += dxy*bcx*bcy

                if (_y < ly - 2) and (_x > 1):
                    Am.append((id0,idp2xp2y))
                    dmat[id0, idm2xp2y] -= dxy
                elif _x > 1:
                    Bm.append((id0,idp2xp2y))
                    dmat[id0, idm2xp2y] -= dxy*bcy
                elif _y < ly - 2:
                    Cm.append((id0,idp2xp2y))
                    dmat[id0, idm2xp2y] -= dxy*bcx
                else:
                    Dm.append((id0,idp2xp2y))
                    dmat[id0, idm2xp2y] -= dxy*bcx*bcy

        print("Ap: ",Ap, '\n', "Bp: ",Bp, '\n', "Cp: ",Cp, '\n', "Dp: ",Dp)
        print("Am: ",Am, '\n', "Bm: ",Bm, '\n', "Cm: ",Cm, '\n', "Dm: ",Dm)

        tmat += tmat.T;    dmat += dmat.T
        # print(np.linalg.eigvalsh(tmat))

        self.ham_real_u = np.block([[tmat, dmat],[dmat, -tmat]])
        self.eng_real_u, self.state_real_u = np.linalg.eigh(self.ham_real_u)

        self.ham_real_d = np.block([[tmat, -dmat],[-dmat, -tmat]])
        self.eng_real_d, self.state_real_d = np.linalg.eigh(self.ham_real_d)

        zmat = np.zeros_like(tmat)

        self.ham_real = np.block([[tmat,           zmat,          zmat,         dmat],
                                  [zmat,           tmat,         -dmat.T,       zmat],
                                  [zmat,          -dmat.conj(),  -tmat.conj(),  zmat],
                                  [dmat.conj().T,  zmat,          zmat,        -tmat.conj()]])
        n = self.Nlat
        u11, u12, u21, u22 = self.state_real_u[:n, :n], self.state_real_u[:n, n:], self.state_real_u[n:, :n], self.state_real_u[n:, n:]
        self.u11, self.u12, self.u21, self.u22 = u11, u12, u21, u22

        self.state_real = np.block([[u11,   zmat,        zmat,        u12],
                                    [zmat,  u22.conj(),  u21.conj(),  zmat],
                                    [zmat,  u12.conj(),  u11.conj(),  zmat],
                                    [u21,   zmat,        zmat,        u22]])

        e = self.state_real.conj().T @ self.ham_real @ self.state_real
        print( np.abs(e-np.diag(np.diag(e))).sum() )
        print( "SU2 eng", np.sort(np.diag(e)) )

        return None

    def calc_real_ham_nonSU2(self):
        """
        a SU(2)-symmetric s-wave BdG model in a nonSU(2) form
        """
        t = self.t
        dxy = self.dxy
        dxxmyy = self.dxxmyy
        print("t=", t, ",  dxy=", dxy, ",  dxxmyy=", dxxmyy)

        bcx, bcy = self.bcx, self.bcy
        N, lx, ly = self.Nlat, self.lx, self.ly

        tmat = np.zeros((N*2, N*2), float)
        dmat = np.zeros((N*2, N*2), float)

        for _x in range(lx):
            for _y in range(ly):
                id0      = 2*self._xy2id(_y,   _x)
                idpx     = 2*self._xy2id(_y,   _x+1)
                idpy     = 2*self._xy2id(_y+1, _x)
                idp2xp2y = 2*self._xy2id(_y+2, _x+2)
                idm2xp2y = 2*self._xy2id(_y+2, _x-2)
                if _x == lx - 1:
                    tmat[id0+0, idpx+0] += t*bcx 
                    tmat[id0+1, idpx+1] += t*bcx 
                    dmat[id0+0, idpx+1] += dxxmyy*bcx 
                    dmat[id0+1, idpx+0] -= dxxmyy*bcx 
                else:
                    tmat[id0+0, idpx+0] += t
                    tmat[id0+1, idpx+1] += t
                    dmat[id0+0, idpx+1] += dxxmyy
                    dmat[id0+1, idpx+0] -= dxxmyy

                if _y == ly - 1:
                    tmat[id0+0, idpy+0] += t*bcy 
                    tmat[id0+1, idpy+1] += t*bcy 
                    dmat[id0+0, idpy+1] -= dxxmyy*bcy 
                    dmat[id0+1, idpy+0] += dxxmyy*bcy 
                else:
                    tmat[id0+0, idpy+0] += t
                    tmat[id0+1, idpy+1] += t 
                    dmat[id0+0, idpy+1] -= dxxmyy
                    dmat[id0+1, idpy+0] += dxxmyy

                if (_y < ly - 2) and (_x < lx - 2):
                    dmat[id0+0, idp2xp2y+1] += dxy
                    dmat[id0+1, idp2xp2y+0] -= dxy
                elif _x < lx - 2:
                    dmat[id0+0, idp2xp2y+1] += dxy*bcy
                    dmat[id0+1, idp2xp2y+0] -= dxy*bcy
                elif _y < ly - 2:
                    dmat[id0+0, idp2xp2y+1] += dxy*bcx
                    dmat[id0+1, idp2xp2y+0] -= dxy*bcx
                else:
                    dmat[id0+0, idp2xp2y+1] += dxy*bcx*bcy
                    dmat[id0+1, idp2xp2y+0] -= dxy*bcx*bcy

                if (_y < ly - 2) and (_x > 1):
                    dmat[id0+0, idm2xp2y+1] -= dxy
                    dmat[id0+1, idm2xp2y+0] += dxy
                elif _x > 1:
                    dmat[id0+0, idm2xp2y+1] -= dxy*bcy
                    dmat[id0+1, idm2xp2y+0] += dxy*bcy
                elif _y < ly - 2:
                    dmat[id0+0, idm2xp2y+1] -= dxy*bcx
                    dmat[id0+1, idm2xp2y+0] += dxy*bcx
                else:
                    dmat[id0+0, idm2xp2y+1] -= dxy*bcx*bcy
                    dmat[id0+1, idm2xp2y+0] += dxy*bcx*bcy


        tmat += tmat.T.conj();    dmat -= dmat.T
        # print(np.linalg.eigvalsh(tmat))
        # print(np.abs(dmat).sum())

        self.ham_real_nonSU2 = np.block([[tmat, dmat],[-dmat.conj(), -tmat.conj()]])
        self.eng_real_nonSU2, self.state_real_nonSU2 = np.linalg.eigh(self.ham_real_nonSU2)

    def calc_real_ham(self, flag):
        if flag == "SU2":
            self.calc_real_ham_SU2()
        else:
            self.calc_real_ham_nonSU2()

    def calc_wannier_state_nonSU2(self):
        print('calculating nonSU(2) g.s.')
        self.calc_real_ham_nonSU2()
        n = self.Nlat*2
        u = self.state_real_nonSU2[:n, :n].T
        v = self.state_real_nonSU2[n:, :n].T
        self.wannier_all_u, self.wannier_all_v = self.Wannier_Z2(u, v)
        return self.wannier_all_u, self.wannier_all_v

    def calc_wannier_state_SU2(self):
        print('calculating SU(2)-symmetric g.s.')
        self.calc_real_ham_SU2()
        u11, u21 = self.u11.T, self.u21.T
        d11, d21 = self.u22.conj().T, self.u12.conj().T
        u11, u21 = self.Wannier_Z2(u11, u21)
        d11, d21 = self.Wannier_Z2(d11, d21)

        n = self.Nlat*2
        u, v = np.zeros((n,n)), np.zeros((n,n))
        for id_ in range(self.Nlat):
            u[id_*2+0, 0::2] = u11[id_, :]
            v[id_*2+0, 1::2] = u21[id_, :]
            u[id_*2+1, 1::2] = d11[id_, :]
            v[id_*2+1, 0::2] = d21[id_, :]
        self.wannier_all_u = u 
        self.wannier_all_v = v
        # M = np.block([[u, v.conj()],[v,u.conj()]])
        # e = M.conj().T @ self.ham_real_nonSU2 @ M
        # print( "hey eee", np.abs(e-np.diag(np.diag(e))).sum() )
        # print( "hey eee", np.sort(np.diag(e)) )
        return self.wannier_all_u, self.wannier_all_v

    def calc_wannier_state(self, flag="SU2"):
        if flag == "SU2":
            u, v = self.calc_wannier_state_SU2()
        elif flag == "SU2PH":
            u, v = self.calc_wannier_state_SU2()
            u = u[0::2, :]
            v = v[0::2, :]
        else:
            u, v = self.calc_wannier_state_nonSU2()
        return u, v

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

    # def plot_path_Gamma_GKMG(self):
    #     x_point = 170
    #     GK = 1/2*np.linspace(0, self.lx, int(x_point/17*10))/2
    #     KM = np.linspace(0, self.lx,int(x_point))/2
    #     MG = np.linspace(self.lx, 0, int(x_point/17*20))/2
    #     engs = []
    #     for _ in range(len(GK)):
    #         e, v = self.calc_k_ham(0, GK[_])
    #         engs.append(e)
    #     for _ in range(len(KM)):
    #         e, v = self.calc_k_ham(KM[_], GK[-1])
    #         engs.append(e)
    #     for _ in range(len(MG)):
    #         e, v = self.calc_k_ham(MG[_], MG[_]/2)
    #         engs.append(e)
    #     plt.plot(np.array(engs))
    #     plt.xticks([0, len(GK),len(GK)+len(KM),len(GK)+len(KM)+len(MG)],[r'$\Gamma$',r'$M$',r'$K$',r'$\Gamma$'])
    #     plt.show()

class FluxSquareLatt(SquareLatt):

    def __init__(self, model_params=dict()):
        super(FluxTriangle, self).__init__(model_params)
        self.Pcf = model_params.get("P", np.inf)
        self.tf  = model_params.get("tf", 0.)
        self.t1  = model_params.get("t1", 1.)
        self.t2  = model_params.get("t2", np.sqrt(0.5))
        self.uf = model_params.get('uf', 0.)
        self.uc = model_params.get('uc', 0.)

    def calc_real_ham(self):
        ''' 
        SU(2) symmetric
        '''

        tf, uf, Pcf = self.tf, self.uf, Pcf
        t1, t2, uc = self.t1, self.t2, self.uc
        N, lx, ly = self.Nlat, self.lx, self.ly
        bcx, bcy = self.bcx, self.bcy

        ham = np.zeros((N*2, N*2), float)

        for _x in range(lx):
            for _y in range(ly):
                id0    = 2*self._xy2id(_y,   _x)
                idpx   = 2*self._xy2id(_y,   _x+1)
                idpy   = 2*self._xy2id(_y+1, _x)
                idpxpy = 2*self._xy2id(_y+1, _x+1)
                idmxpy = 2*self._xy2id(_y+1, _x-1)

                ham[id0+0, id0+0] += uc/2
                ham[id0+1, id0+1] += uf/2
                ham[id0+0, id0+1] += Pcf

                if _y == ly-1 and _x == 0:
                    ham[id0+0, idmxpy+0] = -t2*bcy*bcx
                elif _y == ly-1 and _x !=0:
                    ham[id0+0, idmxpy+0] = -t2*bcy
                elif _x == 0 and _y != ly-1:
                    ham[id0+0, idmxpy+0] = -t2*bcx                
                else:
                    ham[id0+0, idmxpy+0] = -t2          

                if _y == ly-1 and _x == lx-1:
                    ham[id0+0, idpxpy+0] = -t2*bcy*bcx
                elif _y == ly-1 and _x != lx-1:
                    ham[id0+0, idpxpy+0] = -t2*bcy          
                elif _x == lx-1 and _y != ly-1:
                    ham[id0+0, idpxpy+0] = -t2*bcx             
                else:
                    ham[id0+0, idpxpy+0] = -t2

                if _y == ly-1:
                    ham[id0+0, idpy+0] = -t1*bcy
                    ham[id0+1, idpy+1] = +tf*bcy
                else:
                    ham[id0+0, idpy+0] = -t1
                    ham[id0+1, idpy+1] = +tf
                
                if _x == lx-1:
                    ham[id0+0, idpx+0] = -t1*bcx
                    ham[id0+1, idpx+1] = +tf*bcx
                else:
                    ham[id0+0, idpx+0] = -t1
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
            return self.state_real
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
            return self.state_real
        else:
            self.calc_real_ham()
            return self.state_real


