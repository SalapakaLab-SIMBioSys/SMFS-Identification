# =============================================================================
# Description
# =============================================================================

# Class containing methods for pulling experiments Monte-Carlo simulations

# =============================================================================


# =============================================================================
# Imports
# ============================================================================= 
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sympy import var, Eq, solve
from scipy.optimize import root, brentq

# =============================================================================


# =============================================================================
# Class containing methods for pulling experiments Monte-Carlo simulations
# =============================================================================
class MonteCarloSMFS():
    
    
    #--------------------------------------------------------------------------        
    # Init
    #--------------------------------------------------------------------------  
    def __init__(self):         
        # Boltzmann constant
        self.kb = 1.38064852e-23
    #--------------------------------------------------------------------------       
    

    #--------------------------------------------------------------------------
    # find cubic roots
    #--------------------------------------------------------------------------
    def get_cubic_roots(self,a,b,c,d):
        

        a = np.squeeze(np.array([a]))
        b = np.squeeze(np.array([b]))
        c = np.squeeze(np.array([c]))
        d = np.squeeze(np.array([d]))
        
        if(np.size(a) == 1):
            a = np.array([a])
            b = np.array([b])
            c = np.array([c])
            d = np.array([d])
            

        delta0 = b**2 - 3*a*c
        
        delta1 = 2*(b**3) - 9*a*b*c+ 27*(a**2)*d
        
        L_num = delta1 + ((delta1**2) - 4*(delta0**3))**(0.5)
        
        L  = (L_num/2)**(1/3)
        
        z  = -0.5 + 0.5*np.sqrt(3)*1j
        
        L_0_chk = (L==0)
        
        #if(np.sum(L_0_chk) == np.size(L_0_chk)):
        
        x0 = -b/(3*a);
        
        x1 = (-1/(3*a))*(b + L + (delta0/L))
        x2 = (-1/(3*a))*(b + z*L + (delta0/(z*L)))
        x3 = (-1/(3*a))*(b + z*z*L + (delta0/(z*z*L)))
        
        y_shape = np.shape(x1)
        y1 = np.zeros(y_shape)
        y2 = np.zeros(y_shape)
        y3 = np.zeros(y_shape)
        
        y1 = y1.astype('complex128')
        y2 = y2.astype('complex128')
        y3 = y3.astype('complex128')
        
        y1[:] = x1[:]
        y2[:] = x2[:]
        y3[:] = x3[:]
        
        y1[L_0_chk] = x0[L_0_chk]
        y2[L_0_chk] = x0[L_0_chk]
        y3[L_0_chk] = x0[L_0_chk]
        
        cubic_roots = np.array([y1,y2,y3])
        
        
        return(cubic_roots)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    # Solving xp using cubic_solve method
    #--------------------------------------------------------------------------
    def xp_solver_cubic(self,Kf,P,Lc,kb,T,b,Nb):
        c = (kb*T)/(P*Lc)
        
        k3 = 4*(Kf+c)
        k2 = -((9*c*Lc) + (8*Kf*Lc) + (4*Kf*b))
        k1 = (6*Lc*Lc*c) + (4*Kf*Lc*Lc) + (8*Kf*b*Lc)
        k0 = -4*Kf*b*Lc*Lc
        
        xp_sols = self.get_cubic_roots(k3, k2, k1, k0)
        
        xp_sols_check = (np.imag(xp_sols) == 0)
    
        xp = xp_sols[xp_sols_check]
        xp = np.reshape(xp,(Nb,1))
        
        # keep the real part of xp
        xp = np.real(xp)
        
        return xp
    #--------------------------------------------------------------------------
        
    
    #--------------------------------------------------------------------------
    # Generate noise adding to signals
    #--------------------------------------------------------------------------
    def add_noise(self,singal, noise_scale, noise_loc = 0, Noise_existence = True):
        
        if (Noise_existence == True):
            singal_noise = np.random.normal(loc = noise_loc, scale = noise_scale, size = np.shape(singal))
        else:
            singal_noise = np.zeros_like(singal)
            
        singal = singal + singal_noise
        
        return singal
    #--------------------------------------------------------------------------
    

    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    # detachment: simulate the detachment event
    # detachment_method =['variable', 'fixed']. 
        # 'variable': threshold is a gaussian random variable
        # 'fixed': threshold is a constant
    #--------------------------------------------------------------------------
    def detachment(self,rup_no_sep, n_mol, Fwlc_sep, Lc, P, 
                   detachment_prob_value, detachment_thres, detachment_method = 'variable'):
    
        Nb = np.size(rup_no_sep, 0)
        No_column = np.size(rup_no_sep, 1)
        
        # diffrent methods to find detachment_thres 
        if (detachment_method == 'variable'):
            std_multiplier = 1/3
            detachment_thres_mean = detachment_thres#Fwlc_pd.to_numpy().max()
            detachment_thres_std = detachment_thres_mean*std_multiplier
            
            # Assume the detachment force is guassian distributed
            detachment_thres = np.random.normal(detachment_thres_mean,detachment_thres_std,size = Fwlc_sep.shape)
            
        elif(detachment_method == 'fixed'):
            detachment_thres = np.ones_like(Fwlc_sep) * detachment_thres

        # Finding those satisfying detachment requirements
        detachment_event_check = (rup_no_sep == (n_mol + 1)) # all domains unfolded
        detachment_Fwlc_check = (Fwlc_sep > detachment_thres) # larger than detachment_thres
        detachment_check = np.logical_and(detachment_event_check,detachment_Fwlc_check)
        
        detachment_prob = detachment_check * detachment_prob_value 
        r_detach = np.random.rand(Nb,No_column)
        r_detach_check = (r_detach <= detachment_prob)
        
        # Update Lc and P by adding a large number, so that WLC force is almost 0 at 300nm
        # Then we can suppress Fwlc and having xp = ext in the next loop
        dLc_large = 100e-6
        dP_large = 1000e-6#100e-6

        Lc = Lc + np.multiply(r_detach_check, dLc_large) #Multiply arguments element-wise.
        P  = P + np.multiply(r_detach_check, dP_large)
        
        return(Lc,P)
    #--------------------------------------------------------------------------
       
        
    #--------------------------------------------------------------------------
    # Getting WLC force used to solve for xp with fsolve
    #--------------------------------------------------------------------------
    def get_fwlc_fsolve(self, x, Lc, P, b, Kf, T):    
        
        c = (self.kb*T*1e9*1e12)/(P*Lc) # changed to pN*nm for kbT
        a1 = (6*Lc*Lc)-(9*x*Lc)+(4*x*x) 
        a2 = 4*(Lc-x)*(Lc-x)
        Fwlc = (c*x*a1)/a2
        
        Fwlc = np.sum(Fwlc) # summation for multi-column 
        Fwlc = Fwlc - Kf*(b-x)
        
        # Fwlc = Fwlc - Kf*(b-x)
        # Fwlc = np.sum(Fwlc) # summation for multi-column 
        return(Fwlc)
    #-------------------------------------------------------------------------- 
    
    #--------------------------------------------------------------------------
    # Solving for xp fsolve (numerical method to solve for roots)
    #--------------------------------------------------------------------------
    def xp_solver_fsolve(self, Lc, P, T, Kf, b, xp_initial):    
        
        xp_arr = []
        for ii in range(len(Lc)):
            # b_scalar = b[ii][0] * 1e9 # nm
            # Lc_scalar = Lc[ii][0] * 1e9 # nm
            # P_scalar = P[ii][0] * 1e12 # pm
            
            b_scalar = b[ii][0] * 1e9 # nm
            Lc_scalar = Lc[ii] * 1e9 # nm
            P_scalar = P[ii] * 1e12 # pm
            x0 = xp_initial[ii][0]*1e9  # Use previous xp as intial guess
            
            # Solve for xp
            # xp_sols = fsolve(self.get_fwlc_fsolve, x0 = 10, args = (Lc_scalar,P_scalar,b_scalar, Kf, T))
            xp_sols = root(self.get_fwlc_fsolve, x0 = x0, args = (Lc_scalar,P_scalar,b_scalar, Kf, T),tol=1e-10)
            xp_sols = xp_sols.x
            # Ignore image solution
            xp_sols_check = (np.imag(xp_sols) == 0)
        
            xp = xp_sols[xp_sols_check]
            
            # keep the real part of xp
            xp = np.max(np.real(xp))
            
            xp_arr.append(xp*1e-9) # change back to m
        
        xp_arr = np.array(xp_arr)
        
        return(xp_arr)
    #--------------------------------------------------------------------------       
        
    
    #--------------------------------------------------------------------------
    # Use a simple model of adhesive force: triangle singal of Fwlc
    # Modify xp accordingly when adhesion force is on (xp changes slowly during [0, thres_1] and change fast during [thres_1,thres_2])
    #--------------------------------------------------------------------------
    def adhesion_force(self, Fwlc_cur, xp_cur, ad_force_thres, ad_xp_thres = 10e-9, ad_time_thres = 0.1):   
        
        # Randomly find the force when adhesive happens
        ad_force_noise_scale = 3
        ad_force_thres = np.random.normal(loc = ad_force_thres, scale = ad_force_thres/ad_force_noise_scale, size = 1)
        
        # Randomly find the position when adhesive happens
        ad_xp_noise_scale = 3
        ad_xp_thres = np.random.normal(loc = ad_xp_thres, scale = ad_xp_thres/ad_xp_noise_scale, size = 1)
        
        
        # Randomly find the time when adhesive happens
        ad_time_noise_scale = 3
        thres_1 = np.random.normal(loc = ad_time_thres, scale = ad_time_thres/ad_time_noise_scale, size = 1)
        
        if (thres_1<=0):
            thres_1 = ad_time_thres
        
        # Finding the time when adhesive stops
        thres_2 = thres_1*(1+0.05)
        
        # Determining adhesive force
        adhen_force = np.zeros_like(Fwlc_cur)
        data_len = len(Fwlc_cur)
        
        for ii in range(data_len):
            if (ii<=thres_1*data_len):
                adhen_force[ii] = ii/(thres_1*data_len)*ad_force_thres
            elif(ii>thres_1*data_len) and (ii<thres_2*data_len):
                adhen_force[ii] = (thres_2*data_len - ii)/(thres_2*data_len- thres_1*data_len) * ad_force_thres
        
        # Modifying xp
        xp_adhen = np.zeros_like(xp_cur)
        data_len = len(xp_adhen)
        for ii in range(data_len):
            if (ii<=thres_1*data_len):
                xp_adhen[ii] = ii/(thres_1*data_len)*ad_xp_thres
                xp_cur[ii] = 0
            elif(ii>thres_1*data_len) and (ii<thres_2*data_len):
                xp_adhen[ii] = (thres_2*data_len - ii)/(thres_2*data_len - thres_1*data_len) * (ad_xp_thres-xp_cur[int(thres_2*data_len)]) + xp_cur[int(thres_2*data_len)]
                xp_cur[ii] = 0
        xp = xp_cur + xp_adhen
        return (adhen_force, xp)
    #--------------------------------------------------------------------------
    

       
    #--------------------------------------------------------------------------
    # choose DHS methods here
    #--------------------------------------------------------------------------
    def choose_dhs(self, molecule, Energy_shape, kT, Lc):    
        if (Energy_shape == 0):
            # cups like energy landscape
            nu = 1/2
            # Fingerprint:[contour length, dx, k0, dG]
            if (molecule == 'Titin'):
                fingerprint = [Lc, 0.4e-9, 1e-4, 20*kT]
            
            if (molecule == 'UtrNR3'):
                fingerprint = [Lc, 0.375e-9, 0.082, 9.3*kT]
                
            if (molecule == 'UtrNR3_bact'):
                fingerprint = [Lc, 0.85e-9, 0.000876, 14.3*kT]
                
            if (molecule == 'DysNR3_bact'):
                fingerprint = [Lc, 0.6e-9, 0.0111, 11.5*kT]

            # Dudko-Hummer-Szabo (DHS) model, equation (5) in high force catch bond paper
            def rate(rf, dx, k0, dG):
                
                part_same = (1. - (nu * rf * dx/(dG))) # dG is in the unit of kT
                part1 = k0 * part_same**(1./nu - 1.)
                part2 = np.exp((dG/kT) * (1. - part_same**(1./nu)))
                k_off = part1 * part2
                
                return k_off
        
        if (Energy_shape == 1):
            nu = 2/3
            # Fingerprint:[contour length, dx, k0, dG]
            if (molecule == 'Titin'):
                fingerprint = [Lc, 0.34e-9, 2.4e-4, 17.6*kT]
                
            if (molecule == 'UtrNR3'):
                fingerprint = [Lc, 0.33e-9, 0.1054, 8.1*kT]
                    
            if (molecule == 'UtrNR3_bact'):
                fingerprint = [Lc, 0.7e-9, 0.0025, 13*kT]
                
            if (molecule == 'DysNR3_bact'):
                fingerprint = [Lc, 0.48e-9, 0.0302, 10*kT]
                
            # Dudko-Hummer-Szabo (DHS) model, equation (5) in high force catch bond paper
            def rate(rf, dx, k0, dG):
                
                part_same = (1. - (nu * rf * dx/(dG))) # dG is in the unit of kT
                part1 = k0 * part_same**(1./nu - 1.)
                part2 = np.exp((dG/kT) * (1. - part_same**(1./nu)))
                k_off = part1 * part2
                
                return k_off
        return(fingerprint, rate)
    #--------------------------------------------------------------------------  
    
    
# =============================================================================
        
        
        
        