import numpy as np
import scipy as sp
import sys, os
import pickle
from matplotlib import pyplot as plt
from tenpy.tools.params import asConfig


def nxy2siteid(nx, ny, x, y):
    return ((x) % nx) * ny +(y) % ny

class HoneycombLatt():

    def __init__(self, model_params=dict()):
        self.model_params = asConfig(model_params, self.__class__.__name__)
        self.t = model_params.get("t", 1.)
        self.U = model_params.get('U', 10.)
        self.lx = model_params.get("lx", 2)
        self.ly = model_params.get("ly", 2)
        self.Nlat = self.lx * self.ly * 2
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
        self.rab = (self.r1 + self.r2) / 3.

        self.xy2id = dict()
        self.id2xy = dict()
        self.xy2nn = dict()
        self.id2nn = dict()
        self.bcxnn = set()
        self.bcynn = set()
        for _x in range(lx):
            for _y in range(ly):
                xy = (_x, _y)
                idxy = self._xy2id(_y, _x) 
                self.xy2id[xy] = idxy
                self.id2xy[idxy] = xy
                nn0 = self._xy2id(_y, _x) * 2 + 1
                nnx = self._xy2id(_y, _x) * 2 + 0
                nny = self._xy2id(_y, _x+1) * 2 + 0 
                nnz = self._xy2id(_y+1, _x) * 2 + 0
                self.id2nn[idxy] = [nnx, nny, nnz] # note the order of nn
                if _y == ly-1:
                    self.bcynn.add(  tuple(np.sort([nn0, nnz]).tolist()) )
                if _x == lx-1:
                    self.bcxnn.add(  tuple(np.sort([nn0, nny]).tolist()) )

    def calc_real_ham(self):
        tc = 1
        N, lx, ly = self.Nlat, self.lx, self.ly
        bcx, bcy = self.bcx, self.bcy
        
        ham = np.zeros((N, N), float)

        for _x in range(lx):
            for _y in range(ly):
                id0 = self._xy2id(_y,   _x)*2 + 1
                idx = self._xy2id(_y,   _x)*2 + 0
                idy = self._xy2id(_y, _x+1)*2 + 0
                idz = self._xy2id(_y+1, _x)*2 + 0

                ham[id0, idx] += -tc
                if _y == ly - 1:
                    ham[id0, idz] += -tc * bcy
                else:
                    ham[id0, idz] += -tc
                if _x == lx - 1:
                    ham[id0, idy] += -tc * bcx
                else:
                    ham[id0, idy] += -tc

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
        eng, flux_vec = self.calc_real_ham()
        wannier_flux = flux_vec[:, :flux_vec.shape[1]//2]
        wannier_flux = wannier_flux.T
        wannier_flux = self.Wannier_U1(wannier_flux)
        self.wannier_flux = wannier_flux

        N = int(self.Nlat*2); Nmode = int(self.Nlat)

        self.wannier_all = np.zeros((Nmode, N), float)
        # print(self.wannier_flux.shape, self.wannier_all.shape)

        for _ in range(wannier_flux.shape[0]):
            self.wannier_all[_*2+0, 0::2] = wannier_flux[_, :]
            self.wannier_all[_*2+1, 1::2] = wannier_flux[_, :]

        return self.wannier_all, eng

class HoneycombSOC(HoneycombLatt):

    def __init__(self, model_params=dict()):
        super(HoneycombSOC, self).__init__(model_params)
        self.tx = model_params.get("tx", self.t)
        self.ty = model_params.get("ty", self.t)
        self.tz = model_params.get("tz", self.t)

    def calc_real_ham(self):
        tc = self.t
        tx, ty, tz = self.tx, self.ty, self.tz
        N, lx, ly = self.Nlat, self.lx, self.ly
        bcx, bcy = self.bcx, self.bcy
        # print("bcx: ", bcx)
        # print("bcy: ", bcy)

        ham = np.zeros((N*2, N*2), complex)

        for _x in range(lx):
            for _y in range(ly):
                id0u = self._xy2id(_y,   _x)*4 + 2
                id0d = self._xy2id(_y,   _x)*4 + 3
                idxu = self._xy2id(_y,   _x)*4 + 0
                idxd = self._xy2id(_y,   _x)*4 + 1
                idyu = self._xy2id(_y, _x+1)*4 + 0
                idyd = self._xy2id(_y, _x+1)*4 + 1
                idzu = self._xy2id(_y+1, _x)*4 + 0
                idzd = self._xy2id(_y+1, _x)*4 + 1

                # print(id0u, idxu)
                # print(id0d, idxd)
                ham[id0u, idxu] += -tc
                ham[id0d, idxd] += -tc
                ham[id0u, idxd] += -tx
                ham[id0d, idxu] += -tx
                if _y < ly - 1:
                    ham[id0u, idzu] += -tc
                    ham[id0d, idzd] += -tc
                    ham[id0u, idzu] += -tz 
                    ham[id0d, idzd] += +tz
                else:
                    ham[id0u, idzu] += -tc * bcy
                    ham[id0d, idzd] += -tc * bcy
                    ham[id0u, idzu] += -tz * bcy
                    ham[id0d, idzd] += +tz * bcy
                if _x < lx - 1:
                    ham[id0u, idyu] += -tc
                    ham[id0d, idyd] += -tc
                    ham[id0u, idyd] += - -1j*ty 
                    ham[id0d, idyu] += -  1j*ty 
                else:
                    ham[id0u, idyu] += -tc * bcx
                    ham[id0d, idyd] += -tc * bcx
                    ham[id0u, idyd] += - -1j*ty * bcx
                    ham[id0d, idyu] += -  1j*ty * bcx

        ham += ham.T.conj()
        self.ham_real = ham
        self.eng_real, self.state_real = np.linalg.eigh(ham)
        return self.eng_real, self.state_real

    def calc_wannier_state(self):
        eng, flux_vec = self.calc_real_ham()
        wannier_flux = flux_vec[:, :self.Nlat]
        wannier_flux = wannier_flux.T
        wannier_flux = self.Wannier_U1(wannier_flux)
        self.wannier_all = wannier_flux
        return self.wannier_all, eng

class HoneycombKitaev(HoneycombLatt):

    def __init__(self, model_params=dict()):
        super(HoneycombKitaev, self).__init__(model_params)
        self.Jx = model_params.get("Jx", 1.0)
        self.Jy = model_params.get("Jy", 1.0)
        self.Jz = model_params.get("Jz", 1.0)

    def Wannier_Z2(self, g1, g2, N=1):
        norbital, n = g1.shape        
        position = np.power( list(range(1,n+1)), N)
        position = np.diag(position) 
        position12 = g1.conj() @ position @ g1.T + g2.conj() @ position @ g2.T
        position12 = (position12 + position12.T.conj())/2.
        D, U = np.linalg.eigh(position12)
        index = np.argsort(D)
        print(D)
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

    def Kitaev4MF(self, Jxyz=None, bcy=None):
        if bcy is None:
            bcy = self.bcy
        if Jxyz is None:
            Jx, Jy, Jz = np.abs([self.Jx, self.Jy, self.Jz])
        else:
            Jx, Jy, Jz = np.abs(Jxyz)
        nx, ny = self.lx, self.ly
        pn = 1
        n = pn * (nx*ny);  # of interleaved sites

        t = np.zeros((n, n))
        d = np.zeros((n, n))
        for row in range(ny):
            for column in range(nx):
                i0  = nxy2siteid(nx, ny, column, row)
                iy  = nxy2siteid(nx, ny, column+1, row)
                iz  = nxy2siteid(nx, ny, column, row+1)
                t[i0, i0] += Jx/2
                if column < nx-1:
                    t[iy, i0] += Jy/2
                    d[iy, i0] += -Jy/2
                if row < ny-1:
                    t[iz, i0] += Jz/2
                    d[iz, i0] += -Jz/2 
                elif abs(bcy) == 1:
                    t[iz, i0] += Jz/2*bcy
                    d[iz, i0] += -Jz/2*bcy
        t += t.T.conj()
        d -= d.T
        HBdG = np.block([[t, d],[-d.conj(), -t.conj()]])
        energy, M = np.linalg.eigh(HBdG)
        print("energy", energy)
        print(energy[:n].sum()/4)
        V = M[:n,:n].T
        U = M[n:,:n].T
        return V, U

    def KitaevEX(self):
        Jxyz = [self.Jx, self.Jy, self.Jz]
        nx, ny = self.lx, self.ly
        by = self.bcy

        fV, fU = self.Kitaev4MF(Jxyz, by)
        if by == -1:
            v, u = self.Wannier_Z2(fV, fU)
            fV, fU = v, u
        if by == 1:
            v, u = self.Wannier_Z2(fV[:-1,:], fU[:-1,:])
            fV, fU = np.vstack((v, fV[-1,:])), np.vstack((u, fU[-1,:]))

        Jx, Jy, Jz = self.Jx, self.Jy, self.Jz
        a_or_c_x = -np.sign(Jx)
        a_or_c_y = -np.sign(Jy)
        a_or_c_z = -np.sign(Jz)

        pn = 4
        n = pn * (nx*ny);  # of interleaved sites

        print("*******************************generate parton wavefunction*******************************")

        # c^0 Majoranas
        V0, U0 = np.zeros((nx*ny, n), complex ), np.zeros((nx*ny, n), complex )
        count_state = 0
        for nc0 in range(fV.shape[0]):
            for iuc in range(nx*ny):
                V0[count_state, iuc*pn+0] = 1j * (fV[nc0, iuc] + fU[nc0, iuc])/2.
                V0[count_state, iuc*pn+2] = (fV[nc0, iuc] - fU[nc0, iuc])/2.
                U0[count_state, iuc*pn+0] = - 1j * (fV[nc0, iuc] + fU[nc0,iuc])/2.
                U0[count_state, iuc*pn+2] = (- fV[nc0,iuc] + fU[nc0,iuc])/2.
            count_state += 1
        print("construct the u^0 matter fermion with # of 0-degree =", count_state)
        V0 = V0[:count_state, :]
        U0 = U0[:count_state, :]

        # c^x Majoranas
        VX, UX = np.zeros((nx*ny, n), complex ), np.zeros((nx*ny, n), complex )
        count_state = 0
        for row in range(ny):
            for column in range(nx):
                ucll  = nxy2siteid(nx, ny, column, row)*pn
                ucllx = nxy2siteid(nx, ny, column, row)*pn
                VX[count_state, ucll+1] = 0.5
                VX[count_state, ucll+3] = -0.5*1j *a_or_c_x
                UX[count_state, ucll+1] = 0.5
                UX[count_state, ucll+3] = -0.5*1j *a_or_c_x
                count_state += 1;   # move to the next state 
        VX = VX[:count_state, :]
        UX = UX[:count_state, :]
        print("construct the u^x flat band with # of x-degree =", count_state)

        # c^y Majoranas
        VY, UY = np.zeros((nx*ny, n), complex ), np.zeros((nx*ny, n), complex )
        count_state = 0
        for column in range(nx-1):
            for row in range(ny):
                ucll  = nxy2siteid(nx, ny, column, row)*pn
                uclly = nxy2siteid(nx, ny, column+1, row)*pn
                VY[count_state, uclly+1] = -0.5*1j;
                VY[count_state, ucll +3] = -0.5 *a_or_c_y;
                UY[count_state, uclly+1] = 0.5*1j;
                UY[count_state, ucll +3] = 0.5 *a_or_c_y;
                count_state += 1;   # move to the next state 
        if ny % 2 == 0:
            pin_a_or_c_y = 1
            column = 0
            for row in range(ny):
                ucll = ( (column) * ny + row )*pn;
                uclly = ( (column) * ny + (row+1)%ny )*pn;
                AB = 1
                if row%2 == 0:
                    VY[count_state, ucll+AB]  = 0.5*1j ;
                    VY[count_state, uclly+AB] = -0.5*pin_a_or_c_y;
                    UY[count_state, ucll+AB]  = -0.5*1j ;
                    UY[count_state, uclly+AB] = 0.5*pin_a_or_c_y;
                    count_state += 1;   # move to the next state 
            column = nx-1
            for row in range(ny):
                ucll = ( (column) * ny + row )*pn;
                uclly = ( (column) * ny + (row+1)%ny )*pn;
                AB = 3
                if (row%2) == 0:
                    VY[count_state, ucll+AB]  = -0.5*1j ;
                    VY[count_state, uclly+AB] = -0.5*pin_a_or_c_y;
                    UY[count_state, ucll+AB]  = 0.5*1j ;
                    UY[count_state, uclly+AB] = 0.5*pin_a_or_c_y
                    count_state += 1
        else:
            pin_a_or_c_y = 1
            column = 0
            row = 0
            ucll = ( (column) * ny + row )*pn;
            uclly = ( (column) * ny + (row+1)%ny )*pn;
            AB = 1
            VY[count_state, ucll+AB]  = 0.5*1j ;
            VY[count_state, uclly+AB] = -0.5*pin_a_or_c_y;
            UY[count_state, ucll+AB]  = -0.5*1j ;
            UY[count_state, uclly+AB] = 0.5*pin_a_or_c_y;
            count_state += 1;   # move to the next state 

        print("construct the u^y flat band and pin down the boundary c^y Majoranas with # of y-degree =", count_state)
        VY = VY[:count_state, :]
        UY = UY[:count_state, :]

        # c^z Majoranas
        VZ, UZ = np.zeros((nx*ny, n), complex ), np.zeros((nx*ny, n), complex )
        count_state = 0
        for row in range(ny):
            for column in range(nx):
                ucll  = nxy2siteid(nx, ny, column, row)*pn
                ucllz = nxy2siteid(nx, ny, column, row+1)*pn
                if row < ny-1: 
                    VZ[count_state, ucllz+0] = -0.5;
                    VZ[count_state, ucll+2] = 0.5j *a_or_c_z;
                    UZ[count_state, ucllz+0] = -0.5;
                    UZ[count_state, ucll+2] = 0.5j *a_or_c_z;
                    count_state += 1;  
                elif abs(by) == 1:
                    VZ[count_state, ucllz+0] = -0.5;
                    VZ[count_state, ucll+2] = 0.5j*(-1+(by>-1)*2) *a_or_c_z;
                    UZ[count_state, ucllz+0] = -0.5;
                    UZ[count_state, ucll+2] = 0.5j*(-1+(by>-1)*2) *a_or_c_z;
                    count_state += 1;   # move to the next state (u^z band)
        print("construct the u^z flat band with # of z-degree =", count_state)
        # if by == -1:
        #     v, u = self.Wannier_Z2(V0.T, U0.T)
        #     V0, U0 = v, u
        # if by == 1:
        #     v, u = self.Wannier_Z2(V0[:-1,:].T, U0[:-1,:].T)
        #     V0, U0 = np.vstack((v, V0[-1,:])), np.vstack((u, U0[-1,:]))
        VU = [[VX, UX, 1], [VZ, UZ, 0], [VY, UY, 1], [V0, U0, 0]]
        return VU

    def calc_wannier_state(self):
        VU = self.KitaevEX()

        self.wannier_all_u = np.vstack((VU[0][0],VU[1][0],VU[2][0],VU[3][0]))
        self.wannier_all_v = np.vstack((VU[0][1],VU[1][1],VU[2][1],VU[3][1]))

        return self.wannier_all_u, self.wannier_all_v

class FluxHoneycomb(HoneycombLatt):

    def __init__(self, model_params=dict()):
        super(FluxHoneycomb, self).__init__(model_params)
        self.Pcf = model_params.get("P", np.inf)
        self.tf = model_params.get("tf", 0.)
        self.tc = self.t
        self.tx = model_params.get("tx", self.t)
        self.ty = model_params.get("ty", self.t)
        self.tz = model_params.get("tz", self.t)
        self.uf = model_params.get('uf', 0.)
        self.uc = model_params.get('uc', 0.)

    def calc_real_ham(self):
        tc, tf, uc, uf, Pcf = self.tc, self.tf, self.uc, self.uf, self.Pcf
        tx, ty, tz = self.tx, self.ty, self.tz
        N, lx, ly = self.Nlat, self.lx, self.ly
        bcx, bcy = self.bcx, self.bcy

        ham = np.zeros((N*4, N*4), complex)

        for _x in range(lx):
            for _y in range(ly):
                id0 = 8*self._xy2id(_y,  _x)
                for indx in [0, 1, 4, 5]:
                    ham[id0+indx, id0+indx] += uc/2
                for indx in [2, 3, 6, 7]:
                    ham[id0+indx, id0+indx] += uf/2
                for indx in [0, 1, 4, 5]:
                    ham[id0+indx, id0+indx+2] += Pcf

                id0A = 8*self._xy2id(_y,  _x) + 0
                id0B = 8*self._xy2id(_y,  _x) + 4
                idzA = 8*self._xy2id(_y+1,  _x) + 0
                idyA = 8*self._xy2id(_y,  _x+1) + 0

                ham[id0B+0, id0A+0] += -tc
                ham[id0B+1, id0A+1] += -tc
                ham[id0B+0, id0A+1] += -tx
                ham[id0B+1, id0A+0] += -tx
                ham[id0B+2, id0A+2] += +tf
                ham[id0B+3, id0A+3] += +tf

                if _y == ly - 1:
                    ham[id0B+0, idzA+0] += -(tc+tz)*bcy
                    ham[id0B+1, idzA+1] += -(tc-tz)*bcy
                    ham[id0B+2, idzA+2] += +tf*bcy
                    ham[id0B+3, idzA+3] += +tf*bcy
                else:
                    ham[id0B+0, idzA+0] += -(tc+tz)
                    ham[id0B+1, idzA+1] += -(tc-tz)
                    ham[id0B+2, idzA+2] += +tf
                    ham[id0B+3, idzA+3] += +tf
                if _x == lx - 1:
                    ham[id0B+0, idyA+0] += -tc*bcx
                    ham[id0B+1, idyA+1] += -tc*bcx
                    ham[id0B+0, idyA+1] += - -1j*ty*bcx
                    ham[id0B+1, idyA+0] += - +1j*ty*bcx
                    ham[id0B+2, idyA+2] += +tf*bcx
                    ham[id0B+3, idyA+3] += +tf*bcx
                else:
                    ham[id0B+0, idyA+0] += -tc*bcx
                    ham[id0B+1, idyA+1] += -tc*bcx
                    ham[id0B+0, idyA+1] += - -1j*ty*bcx
                    ham[id0B+1, idyA+0] += - +1j*ty*bcx
                    ham[id0B+2, idyA+2] += +tf*bcx
                    ham[id0B+3, idyA+3] += +tf*bcx

        ham += ham.T.conj()
        self.ham_real = ham
        self.eng_real, self.state_real = np.linalg.eigh(ham)
        return self.eng_real, self.state_real

    def calc_wannier_state(self):
        eng, flux_vec = self.calc_real_ham()
        wannier_flux = flux_vec[:, :flux_vec.shape[1]//2]
        wannier_flux = wannier_flux.T
        wannier_flux = self.Wannier_U1(wannier_flux)
        self.wannier_all = wannier_flux
        return self.wannier_all, eng

    def calc_u_real(self, Pcf=None):
        if Pcf is None:
            Pcf = self.Pcf
        if Pcf == np.inf:
            N, lx, ly = self.Nlat, self.lx, self.ly
            self.state_real = np.zeros((N*4, N*2), float)
            rs2 = 1/np.sqrt(2)
            for _ in range(N):
                id0 = _ * 4
                self.state_real[id0+0, _*2+0] = +rs2
                self.state_real[id0+2, _*2+0] = -rs2
                self.state_real[id0+1, _*2+1] = +rs2
                self.state_real[id0+3, _*2+1] = -rs2
            self.state_real = self.state_real.T
            return self.state_real
        elif 1e-12 < abs(self.Pcf) < self.tc:
            print("Warning! ")
            return None
        elif abs(self.Pcf) < 1e-12:
            N, lx, ly = self.Nlat, self.lx, self.ly
            self.state_real = np.zeros((N*4, N*2), complex)
            rs2 = 1/np.sqrt(2)
            paraSOC = dict(t=self.t, tx=self.tx, ty=self.ty, tz=self.tz,
                           lx=self.lx, ly=self.ly, bcx=self.bcx, bcy=self.bcy)
            m = HoneycombSOC(paraSOC)
            m.calc_wannier_state()
            uf = m.wannier_all
            self.state_real = np.zeros((N*4, N*2), complex)
            for _ in range(N//2):
                self.state_real[_*8+2, _*2+0] = +rs2
                self.state_real[_*8+6, _*2+0] = -rs2
                self.state_real[_*8+3, _*2+1] = +rs2
                self.state_real[_*8+7, _*2+1] = -rs2
            for _ in range(uf.shape[0]):
                self.state_real[0::4, N+_] = uf[_, 0::2]
                self.state_real[1::4, N+_] = uf[_, 1::2]
            self.state_real = self.state_real.T
            return self.state_real
        else:
            self.calc_wannier_state()
            return self.wannier_all