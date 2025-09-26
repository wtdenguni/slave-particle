import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

class honeycomb:
    def __init__(self,para):
        self.t = para['t']
        self.b1 = np.array([1,0])*4*np.pi/np.sqrt(3)
        self.b2 = np.array([1/2,np.sqrt(3)/2])*4*np.pi/np.sqrt(3)
        self.a1 = np.array([np.sqrt(3)/2,-1/2])
        self.a2 = np.array([0,1])

    def generate_k(self,N):
        #the total number of k points in N**2
        k1_list,k2_list = np.meshgrid(np.arange(N),np.arange(N))
        k_list = k1_list.flatten()[:,np.newaxis]*self.b1/N + k2_list.flatten()[:,np.newaxis]*self.b2/N
        print(k_list.shape)
        return k_list

    def Fermi_Dirac(self, eg):
        nf = np.zeros_like(eg)
        x = np.exp(-self.beta * np.abs(eg))
        ind = eg < 0.0
        nf[ind] = 1.0 / (1.0 + x)[ind]
        ind = np.logical_not(ind)
        nf[ind] = (x / (1 + x))[ind]

        return nf

    def Bose_Ein(self, eg):
        '''
        eg should always larger than 0
        :param eg:
        :return:
        '''
        nb = np.zeros_like(eg)
        x = np.exp(-self.beta * np.abs(eg))
        ind = eg < 0.0
        nb[ind] = 1.0 / (-1.0 + x)[ind]
        ind = np.logical_not(ind)
        nb[ind] = (x / (1.0 - x))[ind]

        return nb
    
    def mean_field_eq_metallic(self,x0):
        Qf = x0[0]
        Z = x0[1]

        Vk = 1+np.exp(1j*self.kk @ self.a2)+np.exp(-1j*self.kk @ self.a1)
        #spinor part
        #spinor mean field Hamiltonian
        H_f = np.zeros((self.Nk,2,2),dtype = np.complex128)
        H_f[:,0,1] = -self.t*Qf*Vk
        H_f[:,1,0] = -self.t*np.conjugate(Qf*Vk)

        E_f,eig_f = np.linalg.eigh(H_f)

        nf = self.Fermi_Dirac(E_f)
        #calculate the self-consistent QX
        chif_AB = np.einsum('ij,ij->i',np.conjugate(eig_f[:,0,:])*eig_f[:,1,:],nf.reshape((self.Nk,2)))
        QX = (np.sum(Vk*chif_AB)/self.Nk/3)

        #rotor part
        #rotor mean field Hamiltonian
        #determine rho to make the lowest rotor mode gapless(energy = 0)
        H_X = np.zeros((self.Nk,2,2),dtype = np.complex128)
        H_X[:,0,1] = -self.t*np.conjugate(QX)*Vk*2
        H_X[:,1,0] = -self.t*QX*np.conjugate(Vk)*2

        e_X,eig_X = np.linalg.eigh(H_X)
        E_min = np.min(e_X)
        rho = -E_min + 10**(-8)

        H_X = np.zeros((self.Nk,2,2),dtype = np.complex128)
        H_X[:,0,1] = -self.t*np.conjugate(QX)*Vk*2
        H_X[:,1,0] = -self.t*QX*np.conjugate(Vk)*2
        H_X[:,0,0] = rho
        H_X[:,1,1] = rho

        e_X,eig_X = np.linalg.eigh(H_X)
        #calculate the self-consistent Qf and Z
        E_X = np.sqrt(self.U*e_X)
        nX = self.U/2/np.reshape(E_X,(self.Nk,2))*np.reshape(self.Bose_Ein(E_X)-self.Bose_Ein(-E_X),(self.Nk,2))

        #Ek neq 0 part
        chiX_AB = np.einsum('ij,ij->i',np.conjugate(eig_X[1::,1,:])*eig_X[1::,0,:],nX[1::])
        chiX_AA = np.einsum('ij,ij->i',np.conjugate(eig_X[1::,0,:])*eig_X[1::,0,:],nX[1::])

        #Ek = 0 part
        Z_AA = np.abs(eig_X[0,0,0])**2*Z
        Qf_new = (np.sum(Vk[1::]*chiX_AB)/self.Nk/3 +Z*np.conjugate(eig_X[0,1,0])*eig_X[0,0,0])/3*Vk[0]


        print('Z_A=',Z_AA)
        print('Q_f=',Qf_new)
        print('Q_X=',QX)

        ff1 = np.abs(Qf - Qf_new)
        ff2 = Z_AA + np.real(np.sum(chiX_AA)/self.Nk) -1 

        print('f = ',[ff1,ff2])
        print('x0 = ',x0)

        return [ff1,ff2]
    
    def mean_field_eq_Mott(self,x0):
        Qf = x0[0]
        rho = x0[1]

        Vk = 1+np.exp(1j*self.kk @ self.a2)+np.exp(-1j*self.kk @ self.a1)
        #spinor part
        #spinor mean field Hamiltonian
        H_f = np.zeros((self.Nk,2,2),dtype = np.complex128)
        H_f[:,0,1] = -self.t*Qf*Vk
        H_f[:,1,0] = -self.t*np.conjugate(Qf*Vk)
        E_f,eig_f = np.linalg.eigh(H_f)

        nf = self.Fermi_Dirac(E_f)
        #calculate the self-consistent QX
        chif_AB = np.einsum('ij,ij->i',np.conjugate(eig_f[:,0,:])*eig_f[:,1,:],nf.reshape((self.Nk,2)))
        QX = (np.sum(Vk*chif_AB)/self.Nk/3)

        #rotor part
        #rotor mean field Hamiltonian
        H_X = np.zeros((self.Nk,2,2),dtype = np.complex128)
        H_X[:,0,1] = -self.t*np.conjugate(QX)*Vk*2
        H_X[:,1,0] = -self.t*QX*np.conjugate(Vk)*2
        H_X[:,0,0] = rho
        H_X[:,1,1] = rho            

        e_X,eig_X = np.linalg.eigh(H_X)
        print(e_X.shape)
        E_min = np.min(e_X)
        print(E_min)
        E_X = np.sqrt(self.U*e_X)
        nX = self.U/2/np.reshape(E_X,(self.Nk,2))*np.reshape(self.Bose_Ein(E_X)-self.Bose_Ein(-E_X),(self.Nk,2))
        #calculate the self-consistent Qf 
        chiX_AB = np.einsum('ij,ij->i',np.conjugate(eig_X[:,1,:])*eig_X[:,0,:],nX[:])
        chiX_AA = np.einsum('ij,ij->i',np.conjugate(eig_X[:,0,:])*eig_X[:,0,:],nX)

        Qf_new = (np.sum(Vk[:]*chiX_AB)/self.Nk/3)

        print('Q_f=',Qf_new)
        print('Q_X=',QX)

        ff1 = np.abs(Qf - Qf_new)
        ff2 = np.real(np.sum(chiX_AA)/self.Nk) -1.
        print('f = ',[ff1,ff2])
        print('x0 = ',x0)

        return [ff1,ff2]

    def solve_self_consistent_equation(self,U,x0,phase):
        self.beta = 1000

        self.U = U*self.t 

        gd = 100
        self.kk = self.generate_k(gd)   
        self.Nk = self.kk.shape[0]

        if phase == 'metallic':
            #metallic phase returns Qf,Z
            sol = root(self.mean_field_eq_metallic,x0,method = 'hybr')
        elif phase == 'mott':
            #mott phase returns Qf,rho
            sol = root(self.mean_field_eq_Mott,x0,method = 'hybr')
        x0 = sol.x

        return x0


#honeycomb_U = honeycomb({'t':1.0})
#[Qf,rho] = honeycomb_U.solve_self_consistent_equation(1.0)         




        


