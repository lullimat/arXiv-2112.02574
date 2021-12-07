from collabs.TolmanLengthSCMC.LBM_proxy import ShanChenMultiComponent
from collabs.TolmanLengthSCMC.LBM_proxy import CheckCenterOfMassDeltaPConvergence

from idpy.LBM.SCThermo import ShanChanEquilibriumCache, ShanChen
from idpy.LBM.LBM import XIStencils, FStencils, CheckUConvergence, NPT, LBMTypes
from idpy.LBM.LBM import PosFromIndex

from idpy.IdpyCode import GetTenet, GetParamsClean, CheckOCLFP
from idpy.IdpyCode import CUDA_T, OCL_T, idpy_langs_sys

import sympy as sp
import numpy as np
from pathlib import Path
import os
from functools import reduce
from collections import defaultdict
from scipy import interpolate, optimize

'''
Temporary Thermodynamic Variables
'''
n, eps = sp.symbols('n \varepsilon')    
_eps_f = sp.Rational(10, 31)
TLPsis = [sp.exp(-1/n), 1 - sp.exp(-n), 
          ((eps/n + 1) ** (-1 / eps)).subs(eps, _eps_f)]

TLPsiCodes = {TLPsis[0]: 'exp((NType)(-1./ln))', 
              TLPsis[1]: '1. - exp(-(NType)ln)', 
              TLPsis[2]: ('pow(((NType)ln/(' + str(float(_eps_f)) + ' + (NType)ln)), '
                          + str(float(1/_eps_f)) + ')')}

Gcs = {TLPsis[0]: -2.46301869964355,
       TLPsis[1]: -1.3333333333333333}

def AddPosPeriodic(a, b, dim_sizes):
    _swap_add = tuple(map(lambda x, y: x + y, a, b))
    _swap_add = tuple(map(lambda x, y: x + y, _swap_add, dim_sizes))
    _swap_add = tuple(map(lambda x, y: x % y, _swap_add, dim_sizes))
    return _swap_add

def PosNorm2(pos):
    return reduce(lambda x, y: x + y, map(lambda x: x ** 2, pos))

class LatticePressureTensorMC:
    '''
    Ideally, I would need to pass the whole stencil and select the correct lengths
    - Trying to use the general structure of nested lists for the proerties of each
    component
    '''
    def __init__(self, n_field_0 = None, n_field_1 = None,
                 f_stencils = None, psi_syms = None, Gs = None):
        if n_field_0 is None:
            raise Exception("Parameter n_field_0 must not be None")
        if n_field_1 is None:
            raise Exception("Parameter n_field_1 must not be None")
        if f_stencils is None:
            raise Exception("Parameter f_stencils must not be None")
        if psi_syms is None:
            raise Exception("Parameter psi_syms must not be None")
        if Gs is None:
            raise Exception("Parameter Gs must not be None")

        self.n_field_0, self.n_field_1 = n_field_0, n_field_1
        self.n_fields = [self.n_field_0, self.n_field_1]
        
        self.f_stencils, self.psi_syms, self.Gs = f_stencils, psi_syms, Gs        

        self.psi_fs = []
        for _psi_syms_component in self.psi_syms:
            _swap_fs = []
            for _psi_sym in _psi_syms_component:
                _swap_fs += [None if _psi_sym is None else sp.lambdify(n, _psi_sym)]
                
            self.psi_fs += [_swap_fs]

        '''
        Geometric constants
        '''
        self.dim = len(self.n_field_0.shape)
        self.dim_sizes = self.n_field_0.shape
        self.dim_strides = np.array([reduce(lambda x, y: x*y, self.dim_sizes[0:i+1]) 
                                     for i in range(len(self.dim_sizes) - 1)], 
                                    dtype = NPT.C[LBMTypes['SType']])
        self.V = reduce(lambda x, y: x * y, self.dim_sizes)

        '''
        Finding the square-lengths
        '''
        self.l2_lists = []
        for _f_stencils_component in self.f_stencils:
            _l2_list_component_swap = []
            for _f_stencil in _f_stencils_component:
                if _f_stencil is not None:
                    _l2_list_swap = []     
                    for _e in _f_stencil['Es']:
                        _norm2 = PosNorm2(_e)
                        if _norm2 not in _l2_list_swap:
                            _l2_list_swap += [_norm2]
                    _l2_list_component_swap += \
                        [np.array(_l2_list_swap, dtype = np.int32)]
                else:
                    _l2_list_component_swap += [[]]
                        
            self.l2_lists += [_l2_list_component_swap]

        '''
        Init the basis vectors dictionary
        '''
        self.e_l2s = []
        for _l2_lists_component in self.l2_lists:
            _swap_l2s_component = []
            for _l2_list in _l2_lists_component:
                _swap_l2s_component += [{}]
                for l2 in _l2_list:
                    _swap_l2s_component[-1][l2] = []
                    
            self.e_l2s += [_swap_l2s_component]
                    
        _comp_i = 0
        for _f_stencils_component in self.f_stencils:
            _k = 0
            for _f_stencil in _f_stencils_component:
                if _f_stencil is not None:
                    for _e in _f_stencil['Es']:
                        _norm2 = PosNorm2(_e)
                        self.e_l2s[_comp_i][_k][_norm2] += [_e]
                _k += 1
            _comp_i += 1

        '''
        Finding the unique weights
        '''        
        self.w_lists = []
        _comp_i = 0
        for _f_stencils_component in self.f_stencils:
            _k = 0
            _w_lists_swap = []
            for _f_stencil in _f_stencils_component:
                _w_dict, _w_i = {}, 0
                if _f_stencil is not None:
                    _, _swap_idx = np.unique(_f_stencil['Ws'], return_index = True)
                    for l2 in self.l2_lists[_comp_i][_k]:
                        _w_dict[l2] = \
                            np.array(_f_stencil['Ws'])[np.sort(_swap_idx)][_w_i]
                        _w_i += 1
                '''
                Inserting the weights dictionary
                '''
                _w_lists_swap += [_w_dict]
                _k += 1
            '''
            Inserting the list of dictionaries
            '''
            self.w_lists += [_w_lists_swap]
            _comp_i += 1
        
        '''
        Index the lattice pressure tensor contrbutions as a function of the suqare lengths
        '''
        self.pt_groups_f = {1: self.PTLen1, 2: self.PTLen2}

    def PTLen1(self, _pos, _l_psi, l2, _c0, _c1):
        if l2 != 1:
            raise Exception("Parameter l2 must be equal to 1!")
        
        for _ea in self.e_l2s[_c0][_c1][1]:
            #print(_ea)
            '''
            Neighbors
            '''
            _n_ea = tuple(AddPosPeriodic(_pos, np.flip(_ea), self.dim_sizes))
            #print(_pos, np.flip(_ea), _n_ea, self.w_list[1])
            '''
            Need to swap the indices because I need to find the 'other' density
            '''
            _n_psi = self.psi_fields[_c1][_c0][_n_ea]
            _swap_lpt = \
                self.Gs[_c0][_c1] * self.w_lists[_c0][_c1][1] * _l_psi * _n_psi / 2
            
            for i in range(self.dim):
                for j in range(i, self.dim):
                    _lpt_index = (j - i) + self.lpt_dim_strides[i]
                    ##print(j - i, i, self.lpt_dim_strides[i], _lpt_index)
                    self.LPT[(_lpt_index,) +  _pos] += _swap_lpt * _ea[i] * _ea[j]

    def PTLen2(self, _pos, _l_psi, l2, _c0, _c1):
        if l2 != 2:
            raise Exception("Parameter l2 must be equal to 2!")
        
        for _ea in self.e_l2s[_c0][_c1][2]:
            #print(_ea)
            '''
            Neighbors
            '''
            _n_ea = tuple(AddPosPeriodic(_pos, np.flip(_ea), self.dim_sizes))
            #print(_pos, np.flip(_ea), _n_ea, self.w_list[2])
            '''
            Need to swap the indices because I need to find the 'other' density
            '''            
            _n_psi = self.psi_fields[_c1][_c0][_n_ea]
            _swap_lpt = \
                self.Gs[_c0][_c1] * self.w_lists[_c0][_c1][2] * _l_psi * _n_psi / 2
            
            for i in range(self.dim):
                for j in range(i, self.dim):
                    _lpt_index = (j - i) + self.lpt_dim_strides[i]
                    self.LPT[(_lpt_index,) + _pos] += _swap_lpt * _ea[i] * _ea[j]

        
    def GetLPT(self):
        '''
        Declaring geometry-related variables
        '''
        self.n_lpt = self.dim * (self.dim + 1) // 2
        self.lpt_dim_sizes = [self.dim - i for i in range(self.dim - 1)]
        self.lpt_dim_strides = \
            np.array([0] + [reduce(lambda x, y: x + y, self.lpt_dim_sizes[0:i+1]) 
                            for i in range(len(self.lpt_dim_sizes))],
                     dtype = np.int32)
        
        self.LPT = np.zeros([self.n_lpt] + list(self.dim_sizes))
        '''
        Defining the pseudo-potential fields
        '''
        self.psi_fields = []
        _comp_i = 0
        for _psi_fs_component in self.psi_fs:
            _psi_fields_swap = []
            _k = 0
            
            for _psi_f in _psi_fs_component:
                if _psi_f is not None:
                    _psi_fields_swap += [_psi_f(self.n_fields[_comp_i])]
                else:
                    _psi_fields_swap += [[]]

            self.psi_fields += [_psi_fields_swap]
            _comp_i += 1

        for _pos_i in range(self.V):
            _pos = PosFromIndex(_pos_i, self.dim_strides)
            
            '''
            Self interactions
            '''
            if self.psi_fs[0][0] is not None:
                _l_psi_00 = self.psi_fields[0][0][_pos]
                _l_psi_11 = self.psi_fields[1][1][_pos]

                for l2 in self.l2_lists[0][0]:
                    self.pt_groups_f[l2](_pos, _l_psi_00, l2, 0, 0)

                for l2 in self.l2_lists[1][1]:
                    self.pt_groups_f[l2](_pos, _l_psi_11, l2, 1, 1)

            '''
            Inter-components interaction
            '''
            _l_psi_0 = self.psi_fields[0][1][_pos]
            _l_psi_1 = self.psi_fields[1][0][_pos]
            
            for l2 in self.l2_lists[0][1]:
                self.pt_groups_f[l2](_pos, _l_psi_0, l2, 0, 1)

            for l2 in self.l2_lists[1][0]:            
                self.pt_groups_f[l2](_pos, _l_psi_1, l2, 1, 0)
            #sbreak

            '''
            End of interaction pressure tensor
            Adding ideal contribution on diagonal
            '''

        for i in range(self.dim):
            _lpt_index = self.lpt_dim_strides[i]
            self.LPT[_lpt_index, :, :] +=  (self.n_field_0 + self.n_field_1)/3
            
        return self.LPT     
        

class SurfaceOfTensionMC:
    def __init__(self, n_field_0 = None, n_field_1 = None,
                 f_stencils = None, psi_syms = None, Gs = None):
        
        if n_field_0 is None:
            raise Exception("Parameter n_field_0 must not be None")
        if n_field_1 is None:
            raise Exception("Parameter n_field_1 must not be None")
        if f_stencils is None:
            raise Exception("Parameter f_stencils must not be None")
        if psi_syms is None:
            raise Exception("Parameter psi_syms must not be None")
        if Gs is None:
            raise Exception("Parameter Gs must not be None")

        self.n_field_0, self.n_field_1, self.psi_syms, self.Gs = \
            n_field_0, n_field_1, psi_syms, Gs

        self.f_stencils = f_stencils
        
        self.dim = len(self.n_field_0.shape)
        self.dim_center = np.array(list(map(lambda x: x//2, self.n_field_0.shape)))
        self.dim_sizes = self.n_field_0.shape
        '''
        Preparing common variables
        '''
        _LPT_class = LatticePressureTensorMC(self.n_field_0, self.n_field_1,
                                             self.f_stencils, self.psi_syms, self.Gs)
        self.LPT = _LPT_class.GetLPT()

        if self.dim == 2:
            self.r_range = np.arange(self.dim_sizes[1] - self.dim_center[1])
            
            self.radial_n = self.LPT[0, self.dim_center[0], self.dim_center[1]:]
            self.radial_t = self.LPT[2, self.dim_center[0], self.dim_center[1]:]

        if self.dim == 3:
            self.r_range = np.arange(self.dim_sizes[2] - self.dim_center[2])
            
            self.radial_n = self.LPT[0, self.dim_center[0], self.dim_center[1], self.dim_center[2]:]
            self.radial_t = self.LPT[3, self.dim_center[0], self.dim_center[1], self.dim_center[2]:]        

    def GetSurfaceTension(self, grains_fine = 2 ** 10, cutoff = 2 ** 7):
        self.r_fine = np.linspace(self.r_range[0], self.r_range[-1], grains_fine)

        self.radial_t_spl = \
            interpolate.UnivariateSpline(self.r_range, self.radial_t, k = 5, s = 0)
        self.radial_n_spl = \
            interpolate.UnivariateSpline(self.r_range, self.radial_n, k = 5, s = 0)

        '''
        Rowlinson: 4.217
        '''

        def st(R):
            _p_jump = \
                (self.radial_n[0] - (self.radial_n[0] - self.radial_n[-1]) *
                 np.heaviside(self.r_fine - R, 1))

            _swap_spl = \
                interpolate.UnivariateSpline(self.r_fine, 
                                             (self.r_fine ** (self.dim - 1)) *
                                             (_p_jump - self.radial_t_spl(self.r_fine)),
                                             k = 5, s = 0)

            return _swap_spl.integral(self.r_fine[0], self.r_fine[-1]) / (R ** (self.dim - 1))

        _swap_st = np.array([st(rr) for rr in self.r_fine[1:]])
        _swap_st_spl = interpolate.UnivariateSpline(self.r_fine[1:], _swap_st, k = 5, s = 0)
        _swap_rs = optimize.newton(_swap_st_spl.derivative(), x0 = 0)
        _swap_smin = _swap_st_spl(_swap_rs)
        _delta_p_st = self.radial_n[0] - self.radial_n[-1]

        return {'sigma_4.217': _swap_smin,
                'Rs_4.217': _swap_rs,
                'st_spl_4.217': _swap_st_spl,
                'r_fine_4.217': self.r_fine[1:],
                'radial_pt_spl': self.radial_t_spl,
                'radial_pn_spl': self.radial_n_spl,
                'delta_p_st': _delta_p_st}

    def GetSurfaceTensionDerivative(self, grains_fine = 2 ** 10, cutoff = 2 ** 7):
        self.r_fine = np.linspace(self.r_range[0], self.r_range[-1], grains_fine)

        self.radial_t_spl = \
            interpolate.UnivariateSpline(self.r_range, self.radial_t, k = 5, s = 0)
        self.radial_n_spl = \
            interpolate.UnivariateSpline(self.r_range, self.radial_n, k = 5, s = 0)

        '''
        Rowlinson: 4.221
        '''

        def d_st(R):
            _p_jump = \
                (self.radial_n[0] - (self.radial_n[0] - self.radial_n[-1]) *
                 np.heaviside(self.r_fine - R, 1))

            _swap_spl = \
                interpolate.UnivariateSpline(self.r_fine, 
                                             (self.r_fine ** (self.dim - 1)) *
                                             (_p_jump - self.radial_n_spl(self.r_fine)),
                                             k = 5, s = 0)

            return _swap_spl.integral(self.r_fine[0], self.r_fine[-1]) / (R ** (self.dim))

        _swap_d_st = np.array([d_st(rr) for rr in self.r_fine[1:]])
        _swap_d_st_spl = interpolate.UnivariateSpline(self.r_fine[1:], _swap_d_st, k = 5, s = 0)

        return {'d_st_spl_4.221': _swap_d_st_spl, 'r_fine_4.221': self.r_fine[1:]}
    

        
class TolmanSimulationsMC:
    def __init__(self, *args, **kwargs):
        self.InitClass(*args, **kwargs)
        
        self.DumpName()
        '''
        Check if dump exists
        '''
        self.is_there_dump = os.path.isfile(self.dump_name)
        if self.is_there_dump:
            self.full_kwargs = {**self.full_kwargs,
                                **{'empty_sim': True, 'allocate_flag': False}}
            
        self.mc_sim = ShanChenMultiComponent(**self.full_kwargs)

    def End(self):
        self.mc_sim.End()
        del self.mc_sim

    def GetDensityFields(self):
        if self.is_there_dump:
            _n0_swap = \
                self.mc_sim.ReadSnapshotData(file_name = self.dump_name,
                                             full_key =
                                             self.mc_sim.__class__.__name__ + \
                                             '/idpy_memory/n_0')

            _n1_swap = \
                self.mc_sim.ReadSnapshotData(file_name = self.dump_name,
                                             full_key =
                                             self.mc_sim.__class__.__name__ + \
                                             '/idpy_memory/n_1')
        else:
            _n0_swap = self.mc_sim.sims_idpy_memory['n_0'].D2H()
            _n1_swap = self.mc_sim.sims_idpy_memory['n_1'].D2H()

        _n0_swap = _n0_swap.reshape(np.flip(self.mc_sim.sims_vars['dim_sizes']))
        _n1_swap = _n1_swap.reshape(np.flip(self.mc_sim.sims_vars['dim_sizes']))

        return {'n_0': _n0_swap, 'n_1': _n1_swap}

    
    def GetDensityStrips(self, direction = 0):
        _dict_swap = self.GetDensityFields()
        _n0_swap, _n1_swap = _dict_swap['n_0'], _dict_swap['n_1']
        _dim_center = self.mc_sim.sims_vars['dim_center']
        '''
        I will need to get a strip that is as thick as the largest forcing vector(y) (x2)
        '''
        _delta = 1
        if len(self.mc_sim.sims_vars['dim_sizes']) == 2:
            _n0_swap = _n0_swap[_dim_center[1] - _delta:_dim_center[1] + _delta + 1,:]
            _n1_swap = _n1_swap[_dim_center[1] - _delta:_dim_center[1] + _delta + 1,:]

        if len(self.mc_sim.sims_vars['dim_sizes']) == 3:
            _n0_swap = _n0_swap[_dim_center[2] - _delta:_dim_center[2] + _delta + 1,
                                _dim_center[1] - _delta:_dim_center[1] + _delta + 1,:]
            _n1_swap = _n1_swap[_dim_center[2] - _delta:_dim_center[2] + _delta + 1,
                                _dim_center[1] - _delta:_dim_center[1] + _delta + 1,:]
        return {'n_0': _n0_swap, 'n_1': _n1_swap}

    def GetDataDeltaP(self):
        if self.is_there_dump:
            _swap_delta_p = \
                self.mc_sim.ReadSnapshotData(file_name = self.dump_name,
                                             full_key =
                                             self.mc_sim.__class__.__name__ + '/vars/delta_p')
        else:
            _swap_delta_p = self.mc_sim.sims_vars['delta_p']

        _output = {'delta_p': _swap_delta_p[-1]}
        return _output

    '''
    Continue from here
    '''
    def GetDataSurfaceOfTension(self):
        _swap_densities_strips = self.GetDensityStrips()
        _output = {'n_field_0': _swap_densities_strips['n_0'],
                   'n_field_1': _swap_densities_strips['n_1'],
                   'f_stencils': self.params_dict['f_stencils'],
                   'psi_syms': self.params_dict['psi_syms'],
                   'Gs': self.params_dict['SC_Gs']}
        return _output

            
    def Simulate(self, override_flag = False):
        if not self.is_there_dump or override_flag:
            '''
            If we override we also remove the file
            '''
            if override_flag and self.is_there_dump:
                os.remove(self.dump_name)
            '''
            Here we need to retrieve/rerun the flat interface profile
            in order to obtain the equilibrium densities and reach the steady
            state for the droplets faster
            '''
            if self.params_dict['flat_sim_cache']:
                print("Using Flat Interface Simulations for initializing the bulk densities")
                _flat_simulations_params = {'dim_sizes': (127, 5, 5),
                                            'n_components': 2, 
                                            'f_stencils': self.params_dict['f_stencils'],
                                            'n_sum': self.params_dict['n_sum'],
                                            'psi_syms' : self.params_dict['psi_syms'],
                                            'psi_codes': self.params_dict['psi_codes'],
                                            'SC_Gs': self.params_dict['SC_Gs'], 
                                            'taus': self.params_dict['taus'],
                                            'lang': self.params_dict['lang'],
                                            'cl_kind': self.params_dict['cl_kind'], 
                                            'device': self.params_dict['device'], 
                                            'data_dir': self.params_dict['data_dir']}

                _tmp_mc_sim_flat = TolmanSimulationsFlatMC(**_flat_simulations_params)
                _tmp_mc_sim_flat.Simulate()
                _flat_strips = _tmp_mc_sim_flat.GetDensityStrips()
                _n_low_0, _n_high_0 = np.amin(_flat_strips['n_0']), np.amax(_flat_strips['n_0'])
                _n_low_1, _n_high_1 = np.amin(_flat_strips['n_1']), np.amax(_flat_strips['n_1'])
                _tmp_mc_sim_flat.End()
                del _tmp_mc_sim_flat
                print()
            
                self.mc_sim.RadialInterface(n_sum = self.params_dict['n_sum'], 
                                            R = self.params_dict['R'],
                                            invert = self.params_dict['invert'],
                                            n_high_0 = _n_high_0, n_low_0 = _n_low_0,
                                            n_high_1 = _n_high_1, n_low_1 = _n_low_1)
            else:
                self.mc_sim.RadialInterface(n_sum = self.params_dict['n_sum'], 
                                            R = self.params_dict['R'],
                                            invert = self.params_dict['invert'])
                

            self.mc_sim.MainLoopSelfInt(range(0, self.params_dict['max_steps'],
                                              self.params_dict['delta_step']), 
                                        convergence_functions = [CheckUConvergence,
                                                                 CheckCenterOfMassDeltaPConvergence])

            '''
            Check if bubble/droplet burested
            '''
            if abs(self.mc_sim.sims_vars['delta_p'][-1]) < 1e-9:
                print("The", self.params_dict['type'], "has bursted! Dumping Empty simulation")
                '''
                Writing empty simulation file
                '''
                self.mc_sim.sims_dump_idpy_memory_flag = False
                self.mc_sim.sims_vars['empty'] = 'burst'
                self.mc_sim.DumpSnapshot(file_name = self.dump_name,
                                         custom_types = self.mc_sim.custom_types)
                
                return 'burst'
            elif not self.mc_sim.sims_vars['is_centered_seq'][-1]:
                print("The", self.params_dict['type'], "is not centered! Dumping Empty simulation")
                '''
                Writing empty simulation file
                '''
                self.mc_sim.sims_dump_idpy_memory_flag = False
                self.mc_sim.sims_vars['empty'] = 'center'
                self.mc_sim.DumpSnapshot(file_name = self.dump_name,
                                         custom_types = self.mc_sim.custom_types)

                return 'center'
            else:
                print("Dumping in", self.dump_name)
                self.mc_sim.sims_dump_idpy_memory += ['n_0', 'n_1']
                self.mc_sim.DumpSnapshot(file_name = self.dump_name,
                                         custom_types = self.mc_sim.custom_types)
                
            return True
        else:
            print("Dump file", self.dump_name, "already exists!")
            if self.mc_sim.CheckSnapshotData(file_name = self.dump_name,
                                             full_key =
                                             self.mc_sim.__class__.__name__ + '/vars/empty'):

                _swap_val = \
                    self.mc_sim.ReadSnapshotData(file_name = self.dump_name,
                                                 full_key =
                                                 self.mc_sim.__class__.__name__ + '/vars/empty')
                _swap_val = np.array(_swap_val, dtype='<U10')
                print("Empty simulation! Value:", _swap_val)
                return _swap_val
            else:
                return False

        
    def DumpName(self):
        ##print(self.params_dict['f_stencils'])
        _unique_ws = ''
        for _f_stencil_component in self.params_dict['f_stencils']:
            for _f_stencil in _f_stencil_component:
                if _f_stencil is not None:
                    for _w in np.unique(_f_stencil['Ws']):
                        _unique_ws += "%.3f" % _w

        _psi_syms_str = ''
        if 'psi_syms' in self.params_dict:
            for _psi_syms_component in self.params_dict['psi_syms']:
                for _psi_sym in _psi_syms_component:
                    _psi_syms_str += str(_psi_sym)

        _SC_Gs_str = ''
        for _SC_Gs_component in self.params_dict['SC_Gs']:
            for _SC_G in _SC_Gs_component:
                _SC_Gs_str += str(_SC_G)
                
        self.dump_name = \
            (self.__class__.__name__ + '_'
             + str(self.params_dict['dim_sizes']) + '_'
             + 'R' + str(self.params_dict['R']) + '_'
             + str(self.params_dict['type']) + '_'
             + 'Gs' + _SC_Gs_str + '_'
             + 'psis_' + _psi_syms_str + '_'
             + 'ews_' + _unique_ws)

        self.dump_name = self.dump_name.replace("[","_").replace("]","").replace(" ", "_")
        self.dump_name = self.dump_name.replace("/", "_").replace(".","p").replace(",","")
        self.dump_name = self.dump_name.replace("\n", "").replace("(", "").replace(")", "")
        self.dump_name = self.params_dict['data_dir'] / (self.dump_name + '.hdf5')
        
        print(self.dump_name)

    def InitClass(self, *args, **kwargs):
        self.needed_params = ['psi_syms', 'R', 'type', 'n_sum', 'flat_sim_cache',
                              'force_stencils', 'max_steps', 'data_dir', 'ext_sim']
        
        self.needed_params_mc = ['dim_sizes', 'xi_stencil', 'custom_types',
                                 'n_components', 'f_stencils',
                                 'psi_codes', 'SC_Gs',
                                 'taus', 'optimizer_flag',
                                 'lang', 'cl_kind', 'device',
                                 'e2_val']

        
        if not hasattr(self, 'params_dict'):
            self.params_dict = {}
            
        self.kwargs = GetParamsClean(kwargs, [self.params_dict],
                                     needed_params = \
                                     self.needed_params + self.needed_params_mc)

        if 'n_sum' not in self.params_dict:
            raise Exception("Missing argument 'n_sum' (!)")
        
        if 'max_steps' not in self.params_dict:
            self.params_dict['max_steps'] = 2 ** 22

        if 'flat_sim_cache' not in self.params_dict:
            self.params_dict['flat_sim_cache'] = False

        '''
        Merging the dictionaries for passthrough
        '''
        if len(self.params_dict['dim_sizes']) == 2:
            self.params_dict['xi_stencil'] = XIStencils['D2Q9']
            self.params_dict['delta_step'] = 2 ** 14
            if 'data_dir' not in self.params_dict:
                self.params_dict['data_dir'] = Path('data/two-dimensions')

            
        if len(self.params_dict['dim_sizes']) == 3:
            self.params_dict['xi_stencil'] = XIStencils['D3Q19']
            self.params_dict['delta_step'] = 2 ** 11
            if 'data_dir' not in self.params_dict:
                self.params_dict['data_dir'] = Path('data/three-dimensions')


        if ('psi_codes' not in self.params_dict and
            'psi_syms' in self.params_dict):
            self.params_dict['psi_codes'] = []
            for _psi_syms_component in self.params_dict['psi_syms']:
                _psi_codes_swap = []
                for _psi_sym in _psi_syms_component:
                    _psi_codes_swap += [TLPsiCodes[_psi_sym]]
                self.params_dict['psi_codes'] += _psi_codes_swap
            self.params_dict['psi_codes'] = np.array(self.params_dict['psi_codes'])
                    
        self.params_dict['invert'] = \
            True if self.params_dict['type'] == 'A' else False
        
        self.full_kwargs = {**self.kwargs, **self.params_dict}

class TolmanSimulationsFlatMC:
    def __init__(self, *args, **kwargs):
        self.InitClass(*args, **kwargs)
        
        self.DumpName()
        '''
        Check if dump exists
        '''
        self.is_there_dump = os.path.isfile(self.dump_name)
        if self.is_there_dump:
            self.full_kwargs = {**self.full_kwargs,
                                **{'empty_sim': True, 'allocate_flag': False}}
            
        self.mc_sim = ShanChenMultiComponent(**self.full_kwargs)

    def End(self):
        self.mc_sim.End()
        del self.mc_sim

    def GetDensityFields(self):
        if self.is_there_dump:
            _n0_swap = \
                self.mc_sim.ReadSnapshotData(file_name = self.dump_name,
                                             full_key =
                                             self.mc_sim.__class__.__name__ + \
                                             '/idpy_memory/n_0')

            _n1_swap = \
                self.mc_sim.ReadSnapshotData(file_name = self.dump_name,
                                             full_key =
                                             self.mc_sim.__class__.__name__ + \
                                             '/idpy_memory/n_1')
        else:
            _n0_swap = self.mc_sim.sims_idpy_memory['n_0'].D2H()
            _n1_swap = self.mc_sim.sims_idpy_memory['n_1'].D2H()

        _n0_swap = _n0_swap.reshape(np.flip(self.mc_sim.sims_vars['dim_sizes']))
        _n1_swap = _n1_swap.reshape(np.flip(self.mc_sim.sims_vars['dim_sizes']))

        return {'n_0': _n0_swap, 'n_1': _n1_swap}

    
    def GetDensityStrips(self, direction = 0):
        _dict_swap = self.GetDensityFields()
        _n0_swap, _n1_swap = _dict_swap['n_0'], _dict_swap['n_1']
        _dim_center = self.mc_sim.sims_vars['dim_center']
        '''
        I will need to get a strip that is as thick as the largest forcing vector(y) (x2)
        '''
        _delta = 1
        if len(self.mc_sim.sims_vars['dim_sizes']) == 2:
            _n0_swap = _n0_swap[_dim_center[1] - _delta:_dim_center[1] + _delta + 1,:]
            _n1_swap = _n1_swap[_dim_center[1] - _delta:_dim_center[1] + _delta + 1,:]

        if len(self.mc_sim.sims_vars['dim_sizes']) == 3:
            _n0_swap = _n0_swap[_dim_center[2] - _delta:_dim_center[2] + _delta + 1,
                                _dim_center[1] - _delta:_dim_center[1] + _delta + 1,:]
            _n1_swap = _n1_swap[_dim_center[2] - _delta:_dim_center[2] + _delta + 1,
                                _dim_center[1] - _delta:_dim_center[1] + _delta + 1,:]
        return {'n_0': _n0_swap, 'n_1': _n1_swap}

    def GetDataSurfaceTension(self):
        _swap_densities_strips = self.GetDensityStrips()
        _output = {'n_field_0': _swap_densities_strips['n_0'],
                   'n_field_1': _swap_densities_strips['n_1'],
                   'f_stencils': self.params_dict['f_stencils'],
                   'psi_syms': self.params_dict['psi_syms'],
                   'Gs': self.params_dict['SC_Gs']}
        return _output

    def GetFlatSigma(self, grains_fine = 2 ** 10):
        _dim = len(self.mc_sim.sims_vars['dim_sizes'])
        _dim_center = np.flip(self.mc_sim.sims_vars['dim_center'])
        _dim_sizes = np.flip(self.mc_sim.sims_vars['dim_sizes'])

        _dict_get_data_surface_of_tension = self.GetDataSurfaceTension()
        _LPT_class = LatticePressureTensorMC(**_dict_get_data_surface_of_tension)
        _LPT = _LPT_class.GetLPT()

        if _dim == 2:
            _p_n = _LPT[0, _dim_center[0], _dim_center[1]:]
            _p_t = _LPT[2, _dim_center[0], _dim_center[1]:]
            _z_range = np.arange(_dim_sizes[1] - _dim_center[1])            

        if _dim == 3:
            _p_n = _LPT[0, _dim_center[0], _dim_center[1], _dim_center[2]:]
            _p_t = _LPT[3, _dim_center[0], _dim_center[1], _dim_center[2]:]
            _z_range = np.arange(_dim_sizes[2] - _dim_center[2])            

        _p_n_minus_t = _p_n - _p_t
        _p_n_minus_t_spl = interpolate.UnivariateSpline(_z_range,
                                                        _p_n_minus_t,
                                                        k = 5, s = 0)
        _p_n_spl = interpolate.UnivariateSpline(_z_range,
                                                _p_n,
                                                k = 5, s = 0)
        
        _p_t_spl = interpolate.UnivariateSpline(_z_range,
                                                _p_t,
                                                k = 5, s = 0)

        _z_fine = np.linspace(_z_range[0], _z_range[-1], grains_fine)

        
        return {'sigma_flat': _p_n_minus_t_spl.integral(_z_range[0],
                                                        _z_range[-1]),
                'p_n_minus_t_spl': _p_n_minus_t_spl,
                'p_n_spl': _p_n_spl, 'p_t_spl': _p_t_spl,
                'z_fine': _z_fine, 'p_n': _p_n, 'p_t': _p_t,
                **_dict_get_data_surface_of_tension}
    

            
    def Simulate(self):
        if not self.is_there_dump:
            self.mc_sim.FlatInterface(n_sum = self.params_dict['n_sum'],
                                      width = self.mc_sim.sims_vars['dim_sizes'][0] // 2,
                                      direction = 0)
            
            '''
            !!! Remove before publication
            '''
            if self.mc_sim.params_dict['psi_codes'][0][0] is None:
                self.mc_sim.MainLoop(range(0, self.params_dict['max_steps'],
                                           self.params_dict['delta_step']), 
                                     convergence_functions = [CheckUConvergence])

            else:
                self.mc_sim.MainLoopSelfInt(range(0, self.params_dict['max_steps'],
                                                  self.params_dict['delta_step']), 
                                            convergence_functions = [CheckUConvergence])

            print("Dumping in", self.dump_name)
            self.mc_sim.sims_dump_idpy_memory += ['n_0', 'n_1']
            self.mc_sim.DumpSnapshot(file_name = self.dump_name,
                                     custom_types = self.mc_sim.custom_types)

            return True
        else:
            print("Dump file", self.dump_name, "already exists!")
            if self.mc_sim.CheckSnapshotData(file_name = self.dump_name,
                                             full_key =
                                             self.mc_sim.__class__.__name__ + '/vars/empty'):

                _swap_val = \
                    self.mc_sim.ReadSnapshotData(file_name = self.dump_name,
                                                 full_key =
                                                 self.mc_sim.__class__.__name__ + '/vars/empty')
                _swap_val = np.array(_swap_val, dtype='<U10')
                print("Empty simulation! Value:", _swap_val)
                return _swap_val
            else:
                return False

        
    def DumpName(self):
        ##print(self.params_dict['f_stencils'])
        _unique_ws = ''
        for _f_stencil_component in self.params_dict['f_stencils']:
            for _f_stencil in _f_stencil_component:
                if _f_stencil is not None:
                    for _w in np.unique(_f_stencil['Ws']):
                        _unique_ws += "%.3f" % _w

        _psi_syms_str = ''
        if 'psi_syms' in self.params_dict:
            for _psi_syms_component in self.params_dict['psi_syms']:
                for _psi_sym in _psi_syms_component:
                    _psi_syms_str += str(_psi_sym)

        _SC_Gs_str = ''
        for _SC_Gs_component in self.params_dict['SC_Gs']:
            for _SC_G in _SC_Gs_component:
                _SC_Gs_str += str(_SC_G)
                
        self.dump_name = \
            (self.__class__.__name__ + '_'
             + str(self.params_dict['dim_sizes']) + '_'
             + 'Gs' + _SC_Gs_str + '_'
             + 'psis_' + _psi_syms_str + '_'
             + 'ews_' + _unique_ws)

        self.dump_name = self.dump_name.replace("[","_").replace("]","").replace(" ", "_")
        self.dump_name = self.dump_name.replace("/", "_").replace(".","p").replace(",","")
        self.dump_name = self.dump_name.replace("\n", "").replace("(", "").replace(")", "")
        self.dump_name = self.params_dict['data_dir'] / (self.dump_name + '.hdf5')
        
        print(self.dump_name)

    def InitClass(self, *args, **kwargs):
        self.needed_params = ['psi_syms', 'n_sum',
                              'force_stencils', 'max_steps', 'data_dir', 'ext_sim']
        
        self.needed_params_mc = ['dim_sizes', 'xi_stencil', 'custom_types',
                                 'n_components', 'f_stencils',
                                 'psi_codes', 'SC_Gs',
                                 'taus', 'optimizer_flag',
                                 'lang', 'cl_kind', 'device',
                                 'e2_val']

        
        if not hasattr(self, 'params_dict'):
            self.params_dict = {}
            
        self.kwargs = GetParamsClean(kwargs, [self.params_dict],
                                     needed_params = \
                                     self.needed_params + self.needed_params_mc)

        if 'n_sum' not in self.params_dict:
            raise Exception("Missing argument 'n_sum' (!)")
        
        if 'max_steps' not in self.params_dict:
            self.params_dict['max_steps'] = 2 ** 22

        '''
        Merging the dictionaries for passthrough
        '''
        if len(self.params_dict['dim_sizes']) == 2:
            self.params_dict['xi_stencil'] = XIStencils['D2Q9']
            self.params_dict['delta_step'] = 2 ** 14
            if 'data_dir' not in self.params_dict:
                self.params_dict['data_dir'] = Path('data/two-dimensions')

            
        if len(self.params_dict['dim_sizes']) == 3:
            self.params_dict['xi_stencil'] = XIStencils['D3Q19']
            self.params_dict['delta_step'] = 2 ** 11
            if 'data_dir' not in self.params_dict:
                self.params_dict['data_dir'] = Path('data/three-dimensions')


        if ('psi_codes' not in self.params_dict and
            'psi_syms' in self.params_dict):
            self.params_dict['psi_codes'] = []
            for _psi_syms_component in self.params_dict['psi_syms']:
                _psi_codes_swap = []
                for _psi_sym in _psi_syms_component:
                    _psi_codes_swap += [TLPsiCodes[_psi_sym]]
                self.params_dict['psi_codes'] += _psi_codes_swap
            self.params_dict['psi_codes'] = np.array(self.params_dict['psi_codes'])
                            
        self.full_kwargs = {**self.kwargs, **self.params_dict}
