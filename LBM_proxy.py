import numpy as np
from functools import reduce
import sympy as sp

from idpy.IdpyCode import CUDA_T, OCL_T, IDPY_T
from idpy.IdpyCode import GetTenet, GetParamsClean, CheckOCLFP
from idpy.IdpyCode import IdpyMemory

from idpy.IdpyCode.IdpyCode import IdpyKernel, IdpyFunction, IdpyLoop
from idpy.IdpyCode.IdpySims import IdpySims

from idpy.Utils.NpTypes import NpTypes

from idpy.LBM.LBM import RootLB
from idpy.LBM.LBM import InitFStencilWeights, InitDimSizesStridesVolume, InitStencilWeights
from idpy.LBM.LBM import F_IndexFromPos, F_PosFromIndex, F_PointDistanceCenterFirst
from idpy.LBM.LBM import IndexFromPos
from idpy.LBM.LBM import K_InitFlatInterface, K_InitRadialInterface
from idpy.LBM.LBM import K_InitPopulations, F_NFlatProfilePeriodic
from idpy.LBM.LBM import LBMTypes, AllTrue, OneTrue
from idpy.LBM.LBM import K_StreamPeriodic, M_SwapPop, K_ComputePsi
from idpy.LBM.LBM import ComputeCenterOfMass

NPT = NpTypes()

def MCFlatPT_E4(lbm):
    _p_xx, _p_yy = 0, 0
    _direction = lbm.sims_vars['direction']
    _dim_sizes = lbm.sims_vars['dim_sizes']
    _dim_strides = lbm.sims_vars['dim_strides']
    _G = lbm.sims_vars['SC_G_1']
    _c2 = lbm.sims_vars['c2']
    _dim = lbm.sims_vars['DIM']

    _pos = np.copy(lbm.sims_vars['dim_center'])
    _pos[_direction] = 0
    
    _n_0 = lbm.sims_idpy_memory['n_0'].D2H()
    _n_1 = lbm.sims_idpy_memory['n_1'].D2H()
    
    _n_0_swap, _n_1_swap = [], []
    for i in range(_dim_sizes[_direction]):
        _index = IndexFromPos(_pos, _dim_strides)
        _n_0_swap.append(_n_0[_index])
        _n_1_swap.append(_n_1[_index])
        _pos[_direction] += 1
    
    _n_0, _n_1 = np.array(_n_0_swap), np.array(_n_1_swap)
    _n_0_m1 = np.append(_n_0[1:], _n_0[0]) 
    _n_0_p1 = np.append(_n_0[-1], _n_0[:-1])
    
    _n_1_m1 = np.append(_n_1[1:], _n_1[0]) 
    _n_1_p1 = np.append(_n_1[-1], _n_1[:-1])
    
    _p_xx += _c2 * (_n_0 + _n_1)
    _p_xx += _G * _n_0 * (_n_1_p1 + _n_1_m1) / 4
    _p_xx += _G * _n_1 * (_n_0_p1 + _n_0_m1) / 4

    _p_yy += _c2 * (_n_0 + _n_1) + 2 * _G * _n_0 * _n_1 / 3
    _p_yy += _G * _n_0 * (_n_1_p1 + _n_1_m1) / 12
    _p_yy += _G * _n_1 * (_n_0_p1 + _n_0_m1) / 12

    del _n_0, _n_1

    return _p_xx, _p_yy


MultiComponentFlatPT = {'D2E4': MCFlatPT_E4}

def InitRadialInterface(lbm, n_g, n_l, R, full_flag = True, c_i = ''):
    _K_InitRadialInterface = \
        K_InitRadialInterface(custom_types = lbm.custom_types.Push(),
                              constants = lbm.constants,
                              f_classes = [F_PosFromIndex,
                                           F_PointDistanceCenterFirst],
                              optimizer_flag = lbm.optimizer_flag)

    n_g = NPT.C[lbm.custom_types['NType']](n_g)
    n_l = NPT.C[lbm.custom_types['NType']](n_l)
    R = NPT.C[lbm.custom_types['LengthType']](R)
    full_flag = NPT.C[lbm.custom_types['FlagType']](full_flag)

    Idea = _K_InitRadialInterface(tenet = lbm.tenet,
                                  grid = lbm.sims_vars['grid'],
                                  block = lbm.sims_vars['block'])

    Idea.Deploy([lbm.sims_idpy_memory['n' + str(c_i)],
                 lbm.sims_idpy_memory['u'],
                 lbm.sims_idpy_memory['dim_sizes'],
                 lbm.sims_idpy_memory['dim_strides'],
                 lbm.sims_idpy_memory['dim_center'],
                 n_g, n_l, R, full_flag])

    lbm.init_status['n' + str(c_i)] = True
    lbm.init_status['u'] = True
    '''
    Finally, initialize populations...
    Need to do it here: already forgot once to do it outside
    '''
    InitPopulations(lbm, c_i = c_i)


def InitFlatInterface(lbm, n_g, n_l, width, direction = 0,
                      full_flag = True, c_i = ''):
    _K_InitFlatInterface = \
        K_InitFlatInterface(custom_types = lbm.custom_types.Push(),
                            constants = lbm.constants,
                            f_classes = [F_PosFromIndex,
                                         F_NFlatProfilePeriodic],
                            optimizer_flag = lbm.optimizer_flag)

    n_g = NPT.C[lbm.custom_types['NType']](n_g)
    n_l = NPT.C[lbm.custom_types['NType']](n_l)
    width = NPT.C[lbm.custom_types['LengthType']](width)
    direction = NPT.C[lbm.custom_types['SType']](direction)
    full_flag = NPT.C[lbm.custom_types['FlagType']](full_flag)

    Idea = _K_InitFlatInterface(tenet = lbm.tenet,
                                grid = lbm.sims_vars['grid'],
                                block = lbm.sims_vars['block'])

    Idea.Deploy([lbm.sims_idpy_memory['n' + str(c_i)],
                 lbm.sims_idpy_memory['u'],
                 lbm.sims_idpy_memory['dim_sizes'],
                 lbm.sims_idpy_memory['dim_strides'],
                 lbm.sims_idpy_memory['dim_center'],
                 n_g, n_l, width, direction, full_flag])

    lbm.init_status['n' + str(c_i)] = True
    lbm.init_status['u'] = True
    '''
    Finally, initialize populations...
    Need to do it here: already forgot once to do it outside
    '''
    InitPopulations(lbm, c_i = c_i)

def InitPopulations(lbm, c_i = ''):
    if not AllTrue([lbm.init_status['n' + str(c_i)], lbm.init_status['u']]):
        raise Exception("Fields u and n are not initialized")

    _K_InitPopulations = \
        K_InitPopulations(custom_types = lbm.custom_types.Push(),
                          constants = lbm.constants, f_classes = [],
                          optimizer_flag = lbm.optimizer_flag)

    Idea = _K_InitPopulations(tenet = lbm.tenet,
                              grid = lbm.sims_vars['grid'],
                              block = lbm.sims_vars['block'])

    Idea.Deploy([lbm.sims_idpy_memory['pop' + str(c_i)],
                 lbm.sims_idpy_memory['n' + str(c_i)],
                 lbm.sims_idpy_memory['u'],
                 lbm.sims_idpy_memory['XI_list'],
                 lbm.sims_idpy_memory['W_list']])

    lbm.init_status['pop' + str(c_i)] = True

def GridAndBlocks(lbm):
    '''
    looks pretty general
    '''
    _block_size = None
    if 'block_size' in lbm.params_dict:
        _block_size = lbm.params_dict['block_size']
    else:
        _block_size = 128

    _grid = ((lbm.sims_vars['V'] + _block_size - 1)//_block_size, 1, 1)
    _block = (_block_size, 1, 1)

    lbm.sims_vars['grid'], lbm.sims_vars['block'] = _grid, _block

def CheckCenterOfMassDeltaPConvergence(lbm):
    _first_flag = False
    if 'cm_conv' not in lbm.aux_vars:
        lbm.sims_vars['cm_conv'] = []
        lbm.aux_vars.append('cm_conv')

        lbm.sims_vars['cm_coords'] = []
        lbm.aux_vars.append('cm_coords')        

        lbm.sims_vars['delta_p'] = []
        lbm.aux_vars.append('delta_p')

        lbm.sims_vars['p_in'], lbm.sims_vars['p_out'] = \
                        [], []
        lbm.aux_vars.append('p_out')
        lbm.aux_vars.append('p_in')

        lbm.sims_vars['is_centered_seq'] = []
        lbm.aux_vars.append('is_centered_seq')
        
        _first_flag = True
        
    _p_in, _p_out, _delta_p = lbm.DeltaPLaplace()
    print("p_in: ", _p_in, "p_out: ", _p_out, "delta_p: ", _delta_p)
    print()

    _chk, _break_f = [], False
    if not _first_flag:
        _delta_delta_p = _delta_p - lbm.sims_vars['delta_p'][-1]
        _delta_p_in = _p_in - lbm.sims_vars['p_in'][-1]
        _delta_p_out = _p_out - lbm.sims_vars['p_out'][-1]
        
        _chk += [not lbm.sims_vars['is_centered']]
        _chk += [abs(_delta_p) < 1e-9]
        _chk += [abs(_delta_delta_p / _delta_p) < 1e-5]
        #_chk += [abs(_delta_delta_p) < 1e-12]
        #_chk += [abs(_delta_p_in) < 1e-12]
        #_chk += [abs(_delta_p_out) < 1e-12]

        _break_f = OneTrue(_chk)        

        print("Center of mass: ", lbm.sims_vars['cm_coords'])
        print("delta delta_p: ", _delta_delta_p,
              "delta p_in: ", _delta_p_in,
              "delta p_out: ", _delta_p_out)
        
        print(_chk)
        print()

    lbm.sims_vars['cm_conv'].append(np.copy(lbm.sims_vars['cm_coords']))
    lbm.sims_vars['delta_p'].append(_delta_p)
    lbm.sims_vars['p_in'].append(_p_in)
    lbm.sims_vars['p_out'].append(_p_out)
    lbm.sims_vars['is_centered_seq'].append(lbm.sims_vars['is_centered'])
    
    return _break_f

class ShanChenMultiComponent(IdpySims):
    '''
    class ShanChenMultiComponent:
    unfortunately I cannot usee RootLB for inheritance:
    this is a new root class, maybe it is for the best
    '''
    def __init__(self, *args, **kwargs):
        self.SetupRoot(*args, **kwargs)

        self.InitVars()
        GridAndBlocks(self)
        self.InitMemory()
        
        self.init_status = {'u': False}
        for i in range(self.sims_vars['NCOMP']):
            self.init_status['n_' + str(i)] = False
            self.init_status['pop_' + str(i)] = False

    def DeltaPLaplace(self, inside = None, outside = None):
        for i in range(self.sims_vars['NCOMP']):
            if 'n_in_n_out_' + str(i) not in self.sims_idpy_memory:
                self.sims_idpy_memory['n_in_n_out_' + str(i)] = \
                    IdpyMemory.Zeros(2, dtype = NPT.C[self.custom_types['NType']],
                                     tenet = self.tenet)
                self.aux_idpy_memory.append('n_in_n_out_' + str(i))

        _K_NInNOut = K_NInNOut(custom_types = self.custom_types.Push(),
                               constants = self.constants,
                               f_classes = [],
                               optimizer_flag = self.optimizer_flag)

        if inside is None:
            _mass_0, _inside_0 = ComputeCenterOfMass(self, c_i = '_0')
            _inside_0 = list(_inside_0)
            self.sims_vars['cm_coords_0'] = np.array(_inside_0)
            self.sims_vars['mass_0'] = _mass_0
            print("cm_coords_0:", self.sims_vars['cm_coords_0'])
            
            _mass_1, _inside_1 = ComputeCenterOfMass(self, c_i = '_1')
            _inside_1 = list(_inside_1)
            self.sims_vars['cm_coords_1'] = np.array(_inside_1)
            self.sims_vars['mass_1'] = _mass_1
            print("cm_coords_1:", self.sims_vars['cm_coords_1'])            

            """
            for i in range(len(_inside_0)):
                _inside_0[i] = int(round(_inside_0[i]))
                
            for i in range(len(_inside_1)):
                _inside_1[i] = int(round(_inside_1[i]))
            """

            '''
            Check center
            '''
            _chk = False
            for d in range(self.sims_vars['DIM']):
                if abs(self.sims_vars['dim_center'][d] - _inside_0[d]) > 1e-6:
                    _chk = _chk or True

            for d in range(self.sims_vars['DIM']):
                if abs(self.sims_vars['dim_center'][d] - _inside_1[d]) > 1e-6:
                    _chk = _chk or True

            self.sims_vars['is_centered'] = True
            if _chk:
                print("Center of mass (0): ", _inside_0,
                      "Center of mass (1): ", _inside_1,
                      "; Center of the system: ", self.sims_vars['dim_center'])
                self.sims_vars['is_centered'] = False
            
            _inside = IndexFromPos(self.sims_vars['dim_center'].tolist(),
                                   self.sims_vars['dim_strides'])
            _inside = NPT.C['unsigned int'](_inside)
            
        else:
            _inside = NPT.C['unsigned int'](inside)

        if outside is None:
            _outside = self.sims_vars['V'] - 1
            _outside = NPT.C['unsigned int'](_outside)
        else:
            _outside = NPT.C['unsigned int'](outside)

        
        for i in range(self.sims_vars['NCOMP']):
            Idea = _K_NInNOut(tenet = self.tenet,
                              grid = self.sims_vars['grid'],
                              block = self.sims_vars['block'])
            
            Idea.Deploy([self.sims_idpy_memory['n_in_n_out_' + str(i)],
                         self.sims_idpy_memory['n_' + str(i)],
                         _inside, _outside])

        _swap_innout_0 = self.sims_idpy_memory['n_in_n_out_0'].D2H()
        _swap_innout_1 = self.sims_idpy_memory['n_in_n_out_1'].D2H()

        self.sims_vars['n_in_n_out_0'], self.sims_vars['n_in_n_out_1'] = \
            _swap_innout_0, _swap_innout_1
        
        _p_in = self.PBulk(_swap_innout_0[0], _swap_innout_1[0])
        _p_out = self.PBulk(_swap_innout_0[1], _swap_innout_1[1])

        return _p_in, _p_out, _p_in - _p_out

    def PBulk(self, n_a, n_b):
        if 'psi_fs' not in self.sims_vars:
            self.sims_vars['psi_fs'] = []
            for _psi_syms_component in self.sims_vars['psi_syms']:
                _psi_f_comp_swap = []
                for _psi_sym in _psi_syms_component:
                    _psi_f_comp_swap += [sp.lambdify(self.sims_vars['n_sym'],
                                                     _psi_sym if _psi_sym is not None else 0)]

                self.sims_vars['psi_fs'] += [_psi_f_comp_swap]
                
            self.sims_not_dump_vars += ['psi_fs']
        
        _p = ((n_a + n_b) * self.sims_vars['c2'] +
              self.sims_vars['SC_G_0'] * (self.sims_vars['psi_fs'][0][0](n_a) ** 2) / 2 +
              self.sims_vars['SC_G_3'] * (self.sims_vars['psi_fs'][1][1](n_b) ** 2) / 2 +
              self.sims_vars['SC_G_1'] *
              self.sims_vars['psi_fs'][0][1](n_a) *
              self.sims_vars['psi_fs'][1][0](n_b))
        return NPT.C[self.custom_types['NType']](float(_p))

    def MainLoopSelfInt(self, time_steps, convergence_functions = []):
        _all_init = []
        for key in self.init_status:
            _all_init.append(self.init_status[key])

        if not AllTrue(_all_init):
            print(self.init_status)
            raise Exception("Hydrodynamic variables/populations not initialized")
        
        _K_ComputeMomentsMC = K_ComputeMomentsMC(custom_types = self.custom_types.Push(),
                                                 constants = self.constants,
                                                 optimizer_flag = self.optimizer_flag)

        _K_ComputePsi_00 = K_ComputePsi(custom_types = self.custom_types.Push(),
                                        constants = self.constants,
                                        psi_code = self.params_dict['psi_codes'][0][0],
                                        optimizer_flag = self.optimizer_flag)

        _K_ComputePsi_11 = K_ComputePsi(custom_types = self.custom_types.Push(),
                                        constants = self.constants,
                                        psi_code = self.params_dict['psi_codes'][1][1],
                                        optimizer_flag = self.optimizer_flag)        


        _K_Collision_ShanChenGuoMultiComponentSelfInt = \
            K_Collision_ShanChenGuoMultiComponentSelfInt(custom_types = self.custom_types.Push(),
                                                         constants = self.constants,
                                                         f_classes = [F_PosFromIndex,
                                                                      F_IndexFromPos],
                                                         optimizer_flag = self.optimizer_flag)
        
        _K_StreamPeriodic = K_StreamPeriodic(custom_types = self.custom_types.Push(),
                                             constants = self.constants,
                                             f_classes = [F_PosFromIndex,
                                                          F_IndexFromPos],
                                             optimizer_flag = self.optimizer_flag)
        
        self._MainLoop = \
            IdpyLoop(
                [self.sims_idpy_memory],
                [
                    [
                        (_K_ComputeMomentsMC(tenet = self.tenet,
                                           grid = self.sims_vars['grid'],
                                           block = self.sims_vars['block']), ['n_0', 'n_1', 'u',
                                                                              'pop_0', 'pop_1',
                                                                              'XI_list', 'W_list']),

                        (_K_ComputePsi_00(tenet = self.tenet,
                                          grid = self.sims_vars['grid'],
                                          block = self.sims_vars['block']), ['psi_0', 'n_0']),
                        
                        (_K_ComputePsi_11(tenet = self.tenet,
                                          grid = self.sims_vars['grid'],
                                          block = self.sims_vars['block']), ['psi_1', 'n_1']),                        

                        (_K_Collision_ShanChenGuoMultiComponentSelfInt(tenet = self.tenet,
                                                                       grid = self.sims_vars['grid'],
                                                                       block = self.sims_vars['block']),
                         ['pop_0', 'pop_1',
                          'u', 'n_0', 'n_1',
                          'psi_0', 'psi_1',
                          'XI_list', 'W_list',
                          'E_list_1', 'EW_list_1',
                          'E_list_0', 'EW_list_0',
                          'E_list_3', 'EW_list_3',
                          'dim_sizes', 'dim_strides']),
                        
                        (_K_StreamPeriodic(tenet = self.tenet,
                                           grid = self.sims_vars['grid'],
                                           block = self.sims_vars['block']), ['pop_swap', 'pop_0',
                                                                              'XI_list', 'dim_sizes',
                                                                              'dim_strides']),
                        (M_SwapPop(tenet = self.tenet), ['pop_swap', 'pop_0']),

                        (_K_StreamPeriodic(tenet = self.tenet,
                                           grid = self.sims_vars['grid'],
                                           block = self.sims_vars['block']), ['pop_swap', 'pop_1',
                                                                              'XI_list', 'dim_sizes',
                                                                              'dim_strides']),
                        (M_SwapPop(tenet = self.tenet), ['pop_swap', 'pop_1'])
                    ]
                ]
            )

        '''
        now the loop: need to implement the exit condition
        '''
        old_step = 0
        for step in time_steps:
            print(step, step - old_step)
            self._MainLoop.Run(range(step - old_step))
            old_step = step
            if len(convergence_functions):
                checks = []
                for c_f in convergence_functions:
                    checks.append(c_f(self))

                if OneTrue(checks):
                    break
            

    def EqDensities(self, n_sum):
        tau = self.sims_vars['taus'][0]
        G = self.sims_vars['SC_G_1'] * self.constants['CM2']
        '''
        I still need to understand where to multiply by c_s^2...
        '''
                
        def n_g(G):
            return 1/G
        
        _sqrt_in = 45 * (n_g(G) ** 2) - 10 * n_g(G) * n_sum
        _sqrt_in = np.sqrt(_sqrt_in)
        
        _sqrt_out = 30 * (n_g(G) ** 2) - 6 * n_g(G) * _sqrt_in
        _sqrt_out = np.sqrt(_sqrt_out)
        
        _n_high = (n_sum + _sqrt_out)/2
        _n_low = (n_sum - _sqrt_out)/2
        
        return _n_high, _n_low
            

    def FlatInterface(self, n_sum, width, direction = 0, invert = False):
        _full_flag = True if not invert else False

        n_high, n_low = self.EqDensities(n_sum)
        '''
        Record init values
        '''
        self.sims_vars['init_type'] = 'flat'
        self.sims_vars['n_sum'], self.sims_vars['invert'] = n_sum, invert
        self.sims_vars['n_high'], self.sims_vars['n_low'] = n_high, n_low
        self.sims_vars['width'], self.sims_vars['direction'] = width, direction

        for i in range(self.sims_vars['NCOMP']):
            InitFlatInterface(self, n_g = n_low, n_l = n_high,
                              width = width, direction = direction,
                              full_flag = _full_flag, c_i = '_' + str(i))
            ## https://stackoverflow.com/questions/432842/
            ## how-do-you-get-the-logical-xor-of-two-variables-in-python
            _full_flag = _full_flag != True

    def RadialInterface(self, n_sum, R, invert = False,
                        n_high_0 = None, n_low_0 = None,
                        n_high_1 = None, n_low_1 = None):
        
        _full_flag = True if not invert else False

        if (n_high_0 is None or n_low_0 is None or
            n_high_1 is None or n_low_1 is None):
            n_high, n_low = self.EqDensities(n_sum)
            _n_high_list = [n_high, n_high]
            _n_low_list = [n_low, n_low]
            
            self.sims_vars['n_high'], self.sims_vars['n_low'] = n_high, n_low
            self.sims_vars['n_sum'] = n_sum            
        else:
            _n_high_list = [n_high_0, n_high_1]
            _n_low_list = [n_low_0, n_low_1]
            
            self.sims_vars['n_high_0'], self.sims_vars['n_low_0'] = n_high_0, n_low_0
            self.sims_vars['n_high_1'], self.sims_vars['n_low_1'] = n_high_1, n_low_1            
            self.sims_vars['n_sum_0'] = n_high_0 + n_low_0          
            self.sims_vars['n_sum_1'] = n_high_1 + n_low_1
            
        '''
        Record init values
        '''
        self.sims_vars['init_type'] = 'radial'
        self.sims_vars['invert'] = invert
        self.sims_vars['R'] = R

        for i in range(self.sims_vars['NCOMP']):
            InitRadialInterface(self,
                                n_g = _n_low_list[i],
                                n_l = _n_high_list[i],
                                R = R, full_flag = _full_flag,
                                c_i = '_' + str(i))
            ## https://stackoverflow.com/questions/432842/
            ## how-do-you-get-the-logical-xor-of-two-variables-in-python
            _full_flag = _full_flag != True

    def ComputeMoments(self):
        if not AllTrue([self.init_status['pop_' + str(i)]
                        for i in range(self.sims_vars['NCOMP'])]):
            raise Exception("Populations are not initialized")

        _K_ComputeMomentsMC = \
            K_ComputeMomentsMC(custom_types = self.custom_types.Push(),
                               constants = self.constants, f_classes = [],
                               optimizer_flag = self.optimizer_flag)

        Idea = _K_ComputeMomentsMC(tenet = self.tenet,
                                   grid = self.sims_vars['grid'],
                                   block = self.sims_vars['block'])        

        '''
        I can easily parametrize the function call creating the list
        '''
        Idea.Deploy([self.sims_idpy_memory['n_0'],
                     self.sims_idpy_memory['n_1'],
                     self.sims_idpy_memory['u'],
                     self.sims_idpy_memory['pop_0'],
                     self.sims_idpy_memory['pop_1'],
                     self.sims_idpy_memory['XI_list'],
                     self.sims_idpy_memory['W_list']])

        for i in range(self.sims_vars['NCOMP']):
            self.init_status['n_'  + str(i)] = True

        self.init_status['u'] = True


    def InitMemory(self):
        '''
        Init geometrical variables
        '''
        for key in ['dim_sizes', 'dim_strides', 'dim_center', 'XI_list', 'W_list']:
            self.sims_idpy_memory[key] = IdpyMemory.OnDevice(self.sims_vars[key],
                                                             tenet = self.tenet)
        
        for i in range(self.sims_vars['NCOMP']):
            self.sims_idpy_memory['pop_' + str(i)] = \
                IdpyMemory.Zeros(shape = self.sims_vars['V'] * self.sims_vars['Q'],
                                 dtype = NPT.C[self.custom_types['PopType']],
                                 tenet = self.tenet)
            '''
            Two density fields are needed for the equilibrium populations
            '''
            self.sims_idpy_memory['n_' + str(i)] = \
                IdpyMemory.Zeros(shape = self.sims_vars['V'],
                                 dtype = NPT.C[self.custom_types['NType']],
                                 tenet = self.tenet)

            _diag_value = self.sims_vars['diag_comps'][i]
            if self.sims_vars['psi_codes_list'][_diag_value] is not None:
                self.sims_idpy_memory['psi_' + str(i)] = \
                    IdpyMemory.Zeros(shape = self.sims_vars['V'],
                                     dtype = NPT.C[self.custom_types['PsiType']],
                                     tenet = self.tenet)

        '''
        Forcing Stencils
        '''
        for i in range(self.sims_vars['NCOMP']):
            for j in range(self.sims_vars['NCOMP']):
                k = j + i * self.sims_vars['NCOMP']
                if self.sims_vars['E_list_' + str(k)] is not None:
                    self.sims_idpy_memory['E_list_' + str(k)] = \
                        IdpyMemory.OnDevice(self.sims_vars['E_list_' + str(k)],
                                            tenet = self.tenet)

                if self.sims_vars['EW_list_' + str(k)] is not None:
                    self.sims_idpy_memory['EW_list_' + str(k)] = \
                        IdpyMemory.OnDevice(self.sims_vars['EW_list_' + str(k)],
                                            tenet = self.tenet)

        '''
        Only one velocity field is needed for the equilibrium populations
        '''
        self.sims_idpy_memory['u'] = \
            IdpyMemory.Zeros(shape = self.sims_vars['V'] * \
                             self.sims_vars['DIM'],
                             dtype = NPT.C[self.custom_types['UType']],
                             tenet = self.tenet)

        '''
        Using the same number of populations for the components
        '''
        self.sims_idpy_memory['pop_swap'] = \
            IdpyMemory.Zeros(shape = self.sims_vars['V'] * self.sims_vars['Q'],
                             dtype = NPT.C[self.custom_types['PopType']],
                             tenet = self.tenet)


    def InitVars(self):
        '''
        sims_vars
        '''
        self.sims_vars['NCOMP'] = self.params_dict['n_components']
        self.sims_vars['taus'] = self.params_dict['taus']
        self.sims_vars['psi_syms'] = self.params_dict['psi_syms']
        self.sims_not_dump_vars += ['psi_syms']
        
        self.sims_vars['n_sym'] = self.params_dict['n_sym']

        '''
        Here I break the possibility of parametrizing with vectors
        still the idea of using sympy for writing the kernel
        would not need it (?)
        '''
        dim_sizes = self.params_dict['dim_sizes']
        self.sims_vars['DIM'], self.sims_vars['dim_sizes'], \
            self.sims_vars['dim_strides'], self.sims_vars['dim_center'], \
            self.sims_vars['V'] = InitDimSizesStridesVolume(dim_sizes,
                                                            self.custom_types)
        
        self.sims_vars['Q'], self.sims_vars['XI_list'], \
            self.sims_vars['W_list'], self.sims_vars['c2'] = \
                InitStencilWeights(self.params_dict['xi_stencil'],
                                   self.custom_types)    
        
        i_k = 0
        for i in range(self.sims_vars['NCOMP']):
            for _stencil in self.params_dict['f_stencils'][i]:
                if _stencil is not None:
                    self.sims_vars['QE_' + str(i_k)], self.sims_vars['E_list_' + str(i_k)], \
                        self.sims_vars['EW_list_' + str(i_k)] = \
                    InitFStencilWeights(f_stencil = _stencil,
                                        custom_types = self.custom_types)
                else:
                    self.sims_vars['QE_' + str(i_k)], self.sims_vars['E_list_' + str(i_k)], \
                        self.sims_vars['EW_list_' + str(i_k)] = None, None, None
                i_k += 1

        '''
        Here I can manage with None
        '''
        i_k = 0        
        for i in range(self.sims_vars['NCOMP']):
            for _G in self.params_dict['SC_Gs'][i]:
                if _G is not None:
                    self.sims_vars['SC_G_' + str(i_k)] = _G
                else:
                    self.sims_vars['SC_G_' + str(i_k)] = 0
                    
                i_k += 1

        _diag_comps = [i + i * self.sims_vars['NCOMP']
                       for i in range(self.sims_vars['NCOMP'])]
        self.sims_vars['diag_comps'] = np.array(_diag_comps)

        _sc_gs_list = [__ for _ in self.params_dict['SC_Gs'] for __ in _]
        self.sims_vars['SC_Gs_list'] = \
            np.array(_sc_gs_list,
                     dtype = NPT.C[self.custom_types['SCFType']])

        _psi_codes_list = [__ for _ in self.params_dict['psi_codes'] for __ in _]
        self.sims_vars['psi_codes_list'] = np.array(_psi_codes_list)
        '''
        temporary fix to handle the case of 'None' for the hdf5 dump
        '''
        self.sims_not_dump_vars += ['psi_codes_list']
        
        '''
        constants
        '''
        self.constants = {'V': self.sims_vars['V'],
                          'DIM': self.sims_vars['DIM'],
                          'NCOMP': self.sims_vars['NCOMP'],
                          'Q': self.sims_vars['Q'],
                          'CM2': 1/self.sims_vars['c2'],
                          'CM4': 1/(self.sims_vars['c2'] ** 2)}

        '''
        Only the quantities involved in the collision can be parametrized
        '''
        for i in range(self.sims_vars['NCOMP']):
            for j in range(self.sims_vars['NCOMP']):
                _index = j + i * self.sims_vars['NCOMP']
                self.constants['QE_' + str(_index)] = self.sims_vars['QE_' + str(_index)]
                
            self.constants['OMEGA_' + str(i)] = 1./self.params_dict['taus'][i] 

        for i in range(self.sims_vars['NCOMP'] ** 2):
            self.constants['SC_G_' + str(i)] = self.sims_vars['SC_G_' + str(i)]
            if self.constants['SC_G_' + str(i)] is not None:
                self.constants['SC_G_' + str(i)] /= 1

    def SetupRoot(self, *args, **kwargs):
        if not hasattr(self, 'params_dict'):
            self.params_dict = {}

        self.kwargs = GetParamsClean(kwargs, [self.params_dict],
                                     needed_params = ['lang', 'cl_kind', 'device',
                                                      'custom_types', 'block_size',
                                                      'f_stencils',
                                                      'psi_syms', 'psi_codes',
                                                      'SC_Gs', 'n_sym',
                                                      'taus', 'optimizer_flag',
                                                      'n_components', 'xi_stencil',
                                                      'dim_sizes', 'allocate_flag'])
        
        if 'f_stencils' not in self.params_dict:
            raise Exception("Missing 'f_stencils' list")

        if 'xi_stencil' not in self.params_dict:
            raise Exception("Missing 'xi_stencil' parameter")

        if 'taus' not in self.params_dict:
            raise Exception("Missing 'taus' list")
        
        if 'SC_Gs' not in self.params_dict:
            raise Exception("Missing 'SC_Gs' list")

        if 'psi_codes' not in self.params_dict:
            raise Exception("Missing 'psi_codes' list")

        if 'psi_syms' not in self.params_dict:
            raise Exception("Missing 'psi_syms' list")
        
        if 'n_components' not in self.params_dict:
            raise Exception("Parameter 'n_components' is needed")

        if 'lang' not in self.params_dict:
            raise Exception("Param lang = CUDA_T | OCL_T is needed")

        if 'optimizer_flag' in self.params_dict:
            self.optimizer_flag = self.params_dict['optimizer_flag']
        else:
            self.optimizer_flag = True

        if 'allocate_flag' not in self.params_dict:
            self.params_dict['allocate_flag'] = True
            
        '''
        Checking numbers
        '''
        if len(self.params_dict['f_stencils']) != self.params_dict['n_components']:
            raise Exception("Number of forcing stencils differs from number of components!")
        if len(self.params_dict['taus']) != self.params_dict['n_components']:
            raise Exception("Number of taus differs from number of components!")
        if len(self.params_dict['taus']) != self.params_dict['n_components']:
            raise Exception("Number of taus differs from number of components!")

        _matrix_n = 0
        for elem in self.params_dict['SC_Gs']:
            _matrix_n += len(elem)

        if _matrix_n != self.params_dict['n_components'] ** 2:
            raise Exception("Wrong number of entries in the square matrix 'SC_Gs'!")

        _matrix_n = 0
        for elem in self.params_dict['psi_codes']:
            _matrix_n += len(elem)

        if _matrix_n != self.params_dict['n_components'] ** 2:
            raise Exception("Wrong number of entries in the square matrix 'psi_codes'!")

        if 'n_sym' not in self.params_dict:
            self.params_dict['n_sym'] = sp.Symbol('n')

        '''
        Getting Tenet
        '''
        self.tenet = GetTenet(self.params_dict)
        if 'custom_types' in self.params_dict:
            self.custom_types = self.params_dict['custom_types']
        else:
            self.custom_types = LBMTypes

        self.custom_types = \
            CheckOCLFP(tenet = self.tenet, custom_types = self.custom_types)
        
        IdpySims.__init__(self, *args, **self.kwargs)

    def End(self):
        self.tenet.End()


'''
Device Functions
'''
class F_NFlatProfileR(IdpyFunction):
    def __init__(self, custom_types = None, f_type = 'NType'):
        IdpyFunction.__init__(self, custom_types = custom_types, f_type = f_type)
        self.params = {'SType x': ['const'],
                       'LengthType x0': ['const'],
                       'LengthType w0': ['const'],
                       'LengthType w1': ['const']}

        self.functions[IDPY_T] = """
        return tanh((LengthType)(x - (x0 - 0.5 * w0))) - tanh((LengthType)(x - (x0 + 0.5 * w1)));
        """


'''
Kernels
'''
class K_NInNOut(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes)

        self.SetCodeFlags('g_tid')
        self.params = {'NType * n_in_n_out': ['global'],
                       'NType * n': ['global', 'restrict', 'const'],
                       'unsigned int inside': ['const'],
                       'unsigned int outside': ['const']}

        self.kernels[IDPY_T] = """ 
        if(g_tid < V){
            if(g_tid == inside) n_in_n_out[0] = n[inside];
            if(g_tid == outside) n_in_n_out[1] = n[outside];
        }
        """

class K_ComputeMomentsMC(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag)

        self.SetCodeFlags('g_tid')

        self.params = {'NType * n_0': ['global', 'restrict'],
                       'NType * n_1': ['global', 'restrict'],
                       'UType * u': ['global', 'restrict'],
                       'PopType * pop_0': ['global', 'restrict', 'const'],
                       'PopType * pop_1': ['global', 'restrict', 'const'],
                       'SType * XI_list': ['global', 'restrict', 'const'],
                       'WType * W_list': ['global', 'restrict', 'const']}

        self.kernels[IDPY_T] = """
        if(g_tid < V){
            UType lu[DIM];
            for(int d=0; d<DIM; d++){ lu[d] = 0.; }

            NType ln_0 = 0., ln_1 = 0.;
            for(int q=0; q<Q; q++){
                PopType lpop_0 = pop_0[g_tid + q * V];
                PopType lpop_1 = pop_1[g_tid + q * V];
                ln_0 += lpop_0;
                ln_1 += lpop_1;

                for(int d=0; d<DIM; d++){
                    lu[d] += (lpop_0 + lpop_1) * XI_list[d + q * DIM];
                }
            }
            n_0[g_tid] = ln_0;
            n_1[g_tid] = ln_1;
            for(int d=0; d<DIM; d++){ 
                u[g_tid + d * V] = lu[d]/(ln_0 + ln_1);
            }
        }
        """

class K_Collision_ShanChenGuoMultiComponentSelfInt(IdpyKernel):
    def __init__(self, custom_types = {}, constants = {}, f_classes = [],
                 optimizer_flag = None):
        IdpyKernel.__init__(self, custom_types = custom_types,
                            constants = constants, f_classes = f_classes,
                            optimizer_flag = optimizer_flag)
        self.SetCodeFlags('g_tid')
        self.params = {'PopType * pop_0': ['global', 'restrict'],
                       'PopType * pop_1': ['global', 'restrict'],
                       'UType * u': ['global', 'restrict'],
                       'NType * n_0': ['global', 'restrict', 'const'],
                       'NType * n_1': ['global', 'restrict', 'const'],
                       'PsiType * psi_0': ['global', 'restrict', 'const'],
                       'PsiType * psi_1': ['global', 'restrict', 'const'],
                       'SType * XI_list': ['global', 'restrict', 'const'],
                       'WType * W_list': ['global', 'restrict', 'const'],
                       'SType * E_list': ['global', 'restrict', 'const'],
                       'WType * EW_list': ['global', 'restrict', 'const'],
                       'SType * E_list_00': ['global', 'restrict', 'const'],
                       'WType * EW_list_00': ['global', 'restrict', 'const'],
                       'SType * E_list_11': ['global', 'restrict', 'const'],
                       'WType * EW_list_11': ['global', 'restrict', 'const'],
                       'SType * dim_sizes': ['global', 'restrict', 'const'],
                       'SType * dim_strides': ['global', 'restrict', 'const']}
        
        self.kernels[IDPY_T] = """
        if(g_tid < V){
            // Getting local densities
            NType ln_0 = n_0[g_tid], ln_1 = n_1[g_tid];
            // Getting thread position
            SType g_tid_pos[DIM];
            F_PosFromIndex(g_tid_pos, dim_sizes, dim_strides, g_tid);

            // Computing Shan-Chen Force
            SCFType F_0[DIM], F_1[DIM], F_00[DIM], F_11[DIM]; SType neigh_pos[DIM];
            for(int d=0; d<DIM; d++){F_0[d] = F_1[d] = F_00[d] = F_11[d] = 0.;}

            PsiType lpsi_0 = psi_0[g_tid], lpsi_1 = psi_1[g_tid];

            // Inter-Component
            for(int qe=0; qe<QE_1; qe++){
                // Compute neighbor position
                for(int d=0; d<DIM; d++){
                    neigh_pos[d] = ((g_tid_pos[d] + E_list[d + qe*DIM] + dim_sizes[d]) % dim_sizes[d]);
                }
                // Compute neighbor index
                SType neigh_index = F_IndexFromPos(neigh_pos, dim_strides);
                // Get the pseudopotential value
                PsiType nn_1 = n_1[neigh_index], nn_0 = n_0[neigh_index];
                // Add partial contribution
                for(int d=0; d<DIM; d++){
                    F_0[d] += E_list[d + qe*DIM] * EW_list[qe] * nn_1; 
                    F_1[d] += E_list[d + qe*DIM] * EW_list[qe] * nn_0;
                }
            }

            // Self Component 0
            for(int qe=0; qe<QE_0; qe++){
                // Compute neighbor position
                for(int d=0; d<DIM; d++){
                    neigh_pos[d] = ((g_tid_pos[d] + E_list_00[d + qe*DIM] + dim_sizes[d]) % dim_sizes[d]);
                }
                // Compute neighbor index
                SType neigh_index = F_IndexFromPos(neigh_pos, dim_strides);
                // Get the pseudopotential value
                PsiType npsi_0 = psi_0[neigh_index];
                // Add partial contribution
                for(int d=0; d<DIM; d++){
                    F_00[d] += E_list_00[d + qe*DIM] * EW_list_00[qe] * npsi_0; 
                }
            }

            // Self Component 1
            for(int qe=0; qe<QE_3; qe++){
                // Compute neighbor position
                for(int d=0; d<DIM; d++){
                    neigh_pos[d] = ((g_tid_pos[d] + E_list_00[d + qe*DIM] + dim_sizes[d]) % dim_sizes[d]);
                }
                // Compute neighbor index
                SType neigh_index = F_IndexFromPos(neigh_pos, dim_strides);
                // Get the pseudopotential value
                PsiType npsi_1 = psi_1[neigh_index];
                // Add partial contribution
                for(int d=0; d<DIM; d++){
                    F_11[d] += E_list_11[d + qe*DIM] * EW_list_11[qe] * npsi_1;
                }
            }

            for(int d=0; d<DIM; d++){
                F_0[d] *= -SC_G_1 * ln_0;
                F_1[d] *= -SC_G_2 * ln_1;
                // Here we sum the self-interactions
                // the rest of the algorithm should not change
                F_0[d] += -SC_G_0 * lpsi_0 * F_00[d];
                F_1[d] += -SC_G_3 * lpsi_1 * F_11[d];
            }

            // Local density and velocity for Guo velocity shift and equilibrium
            UType lu[DIM];

            // Guo velocity shift & Copy to global memory
            for(int d=0; d<DIM; d++){ 
                lu[d] = u[g_tid + V*d] + 0.5 * (F_0[d] + F_1[d])/(ln_0 + ln_1);
                u[g_tid + V*d] = lu[d];
            }

            // Compute square norm of Guo shifted velocity
            UType u_dot_u = 0.;
            for(int d=0; d<DIM; d++){
                u_dot_u += lu[d]*lu[d];
            }

            // Cycle over the populations: equilibrium + Guo
            for(int q=0; q<Q; q++){
                UType u_dot_xi = 0.; 
                UType F_dot_xi_0 = 0., F_dot_u_0 = 0.; 
                UType F_dot_xi_1 = 0., F_dot_u_1 = 0.; 
                for(int d=0; d<DIM; d++){
                    u_dot_xi += lu[d] * XI_list[d + q*DIM];

                    F_dot_xi_0 += F_0[d] * XI_list[d + q*DIM];
                    F_dot_u_0  += F_0[d] * lu[d];
                    F_dot_xi_1 += F_1[d] * XI_list[d + q*DIM];
                    F_dot_u_1  += F_1[d] * lu[d];
                }

                PopType leq_pop = 1.;
                PopType leq_pop_0 = 1., lguo_pop_0 = 0.;
                PopType leq_pop_1 = 1., lguo_pop_1 = 0.;

                // Equilibrium populations
                leq_pop += + u_dot_xi*CM2 + 0.5*u_dot_xi*u_dot_xi*CM4;
                leq_pop += - 0.5*u_dot_u*CM2;
                leq_pop = leq_pop * W_list[q];

                leq_pop_0 = leq_pop * ln_0;
                leq_pop_1 = leq_pop * ln_1;

                // Guo populations
                lguo_pop_0 += + F_dot_xi_0*CM2 + F_dot_xi_0*u_dot_xi*CM4;
                lguo_pop_0 += - F_dot_u_0*CM2;
                lguo_pop_0 = lguo_pop_0 * W_list[q];

                lguo_pop_1 += + F_dot_xi_1*CM2 + F_dot_xi_1*u_dot_xi*CM4;
                lguo_pop_1 += - F_dot_u_1*CM2;
                lguo_pop_1 = lguo_pop_1 * W_list[q];

                pop_0[g_tid + q*V] = \
                    pop_0[g_tid + q*V]*(1. - OMEGA_0) + leq_pop_0*OMEGA_0 + \
                    (1. - 0.5 * OMEGA_0) * lguo_pop_0;

                pop_1[g_tid + q*V] = \
                    pop_1[g_tid + q*V]*(1. - OMEGA_1) + leq_pop_1*OMEGA_1 + \
                    (1. - 0.5 * OMEGA_1) * lguo_pop_1;

             }
        }
        """
