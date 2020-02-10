from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from OpenGoddard.optimize import Condition, Dynamics
import time
from functools import partial
import os
import datetime
from scipy.integrate import solve_ivp
import sys
from multiprocessing import Pool
from scipy import special
from scipy import interpolate
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splev, splrep
from scipy.optimize import basinhopping, Bounds
import scipy.io as sio
from import_initialCond import init_conds

sys.path.insert(0, 'home/francesco/git_workspace/FESTIP_Work')


'''this script is a collocation algorithm on FESTIP model'''

class Problem:
    """ OpenGoddard Problem class.
    Args:
        time_init (list of float) : [time_start, time_section0, time_section0, , , time_final]
        nodes (int) : number of nodes
        number_of_states (list) : number of states
        number_of_controls (list) : number of controls
        maxIterator (int) : iteration max
    Attributes:
        nodes (int) : time nodes.
        number_of_states (int) : number of states.
        number_of_controls (int) : number of controls
        number_of_section (int) : number of section
        number_of_param (int) : number of inner variables
        div (list) : division point of inner variables
        tau : Gauss nodes
        w : weights of Gaussian quadrature
        D :  differentiation matrix of Gaussian quadrature
        time : time
        maxIterator (int) : max iterator
        time_all_section : all section time
        unit_states (list of float) : canonical unit of states
        unit_controls (list of float) : canonical unit of controls
        unit_time (float) : canonical unit of time
        p ((N,) ndarray) : inner variables for optimization
        dynamics (function) : function list, list of function of dynamics
        knot_states_smooth (list of True/False): list of states are smooth on phase knots
        cost (function) : cost function
        running_cost (function, optional) : (default = None)
        cost_derivative (function, optional) : (default = None)
        equality (function) : (default = None)
        inequality (function) : (default = None)
    """
    def _LegendreFunction(self, x, n):
        Legendre, Derivative = special.lpn(n, x)
        return Legendre[-1]

    def _LegendreDerivative(self, x, n):
        Legendre, Derivative = special.lpn(n, x)
        return Derivative[-1]

    def _nodes_LG(self, n):
        '''Return Gauss-Legendre nodes.'''
        nodes, weight = special.p_roots(n)
        return nodes

    def _weight_LG(self, n):
        '''Return Gauss-Legendre weight.'''
        nodes, weight = special.p_roots(n)
        return weight

    def _differentiation_matrix_LG(self, n):
        tau = self._nodes_LG(n)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i, j] = self._LegendreDerivative(tau[i], n) \
                              / self._LegendreDerivative(tau[j], n) \
                              / (tau[i] - tau[j])
                else:
                    D[i, j] = tau[i] / (1 - tau[i]**2)
        return D

    def method_LG(self, n):
        """ Legendre-Gauss Pseudospectral method
        Gauss nodes are roots of :math:`P_n(x)`.
        Args:
            n (int) : number of nodes
        Returns:
            ndarray, ndarray, ndarray : nodes, weight, differentiation_matrix
        """
        nodes, weight = special.p_roots(n)
        D = _differentiation_matrix_LG(n)
        return nodes, weight, D

    def _nodes_LGR(self, n):
        '''Return Gauss-Radau nodes.'''
        roots, weight = special.j_roots(n-1, 0, 1)
        nodes = np.hstack((-1, roots))
        return nodes

    def _weight_LGR(self, n):
        '''Return Gauss-Legendre weight.'''
        nodes = self._nodes_LGR(n)
        w = np.zeros(0)
        for i in range(n):
            w = np.append(w, (1-nodes[i])/(n*n*self._LegendreFunction(nodes[i], n-1)**2))
        return w

    def _differentiation_matrix_LGR(self, n):
        tau = self._nodes_LGR(n)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i, j] = self._LegendreFunction(tau[i], n-1) \
                              / self._LegendreFunction(tau[j], n-1) \
                              * (1 - tau[j]) / (1 - tau[i]) / (tau[i] - tau[j])
                elif i == j and i == 0:
                    D[i, j] = -(n-1)*(n+1)*0.25
                else:
                    D[i, j] = 1 / (2 * (1 - tau[i]))
        return D

    def method_LGR(self, n):
        """ Legendre-Gauss-Radau Pseudospectral method
        Gauss-Radau nodes are roots of :math:`P_n(x) + P_{n-1}(x)`.
        Args:
            n (int) : number of nodes
        Returns:
            ndarray, ndarray, ndarray : nodes, weight, differentiation_matrix
        """
        nodes = _nodes_LGR(n)
        weight = _weight_LGR(n)
        D = _differentiation_matrix_LGR(n)
        return nodes, weight, D

    def _nodes_LGL_old(self, n):
        """Return Legendre-Gauss-Lobatto nodes.
        Gauss-Lobatto nodes are roots of P'_{n-1}(x) and -1, 1.
        ref. http://keisan.casio.jp/exec/system/1360718708
        """
        x0 = np.zeros(0)
        for i in range(2, n):
            xi = (1-3.0*(n-2)/8.0/(n-1)**3)*np.cos((4.0*i-3)/(4.0*(n-1)+1)*np.pi)
            x0 = np.append(x0, xi)
        x0 = np.sort(x0)

        roots = np.zeros(0)
        for x in x0:
            optResult = optimize.root(self._LegendreDerivative, x, args=(n-1,))
            roots = np.append(roots, optResult.x)
        nodes = np.hstack((-1, roots, 1))
        return nodes

    def _nodes_LGL(self, n):
        """ Legendre-Gauss-Lobatto(LGL) points"""
        roots, weight = special.j_roots(n-2, 1, 1)
        nodes = np.hstack((-1, roots, 1))
        return nodes

    def _weight_LGL(self, n):
        """ Legendre-Gauss-Lobatto(LGL) weights."""
        nodes = self._nodes_LGL(n)
        w = np.zeros(0)
        for i in range(n):
            w = np.append(w, 2/(n*(n-1)*self._LegendreFunction(nodes[i], n-1)**2))
        return w

    def _differentiation_matrix_LGL(self, n):
        """ Legendre-Gauss-Lobatto(LGL) differentiation matrix."""
        tau = self._nodes_LGL(n)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i, j] = self._LegendreFunction(tau[i], n-1) \
                              / self._LegendreFunction(tau[j], n-1) \
                              / (tau[i] - tau[j])
                elif i == j and i == 0:
                    D[i, j] = -n*(n-1)*0.25
                elif i == j and i == n-1:
                    D[i, j] = n*(n-1)*0.25
                else:
                    D[i, j] = 0.0
        return D

    def method_LGL(self, n):
        """ Legendre-Gauss-Lobatto Pseudospectral method
        Gauss-Lobatto nodes are roots of :math:`P'_{n-1}(x)` and -1, 1.
        Args:
            n (int) : number of nodes
        Returns:
            ndarray, ndarray, ndarray : nodes, weight, differentiation_matrix
        References:
            Fariba Fahroo and I. Michael Ross. "Advances in Pseudospectral Methods
            for Optimal Control", AIAA Guidance, Navigation and Control Conference
            and Exhibit, Guidance, Navigation, and Control and Co-located Conferences
            http://dx.doi.org/10.2514/6.2008-7309
        """
        nodes = _nodes_LGL(n)
        weight = _weight_LGL(n)
        D = _differentiation_matrix_LGL(n)
        return nodes, weight, D

    def _make_param_division(self, nodes, number_of_states, number_of_controls):
        prev = 0
        div = []
        for index, node in enumerate(nodes):
            num_param = number_of_states[index] + number_of_controls[index]
            temp = [i*(node) + prev for i in range(1, num_param + 1)]
            prev = temp[-1]
            div.append(temp)
        return div

    def _division_states(self, state, section):
        assert section < len(self.nodes), \
            "section argument out of own section range"
        assert state < self.number_of_states[section], \
            "states argument out of own states range"
        if (state == 0):
            if (section == 0):
                div_front = 0
            else:
                div_front = self.div[section-1][-1]
        else:
            div_front = self.div[section][state-1]
        div_back = self.div[section][state]
        return div_back, div_front

    def _division_controls(self, control, section):
        assert section < len(self.nodes), \
            "section argument out of own section range"
        assert control < self.number_of_controls[section], \
            "controls argument out of own controls range"
        div_front = self.div[section][self.number_of_states[section] + control - 1]
        div_back = self.div[section][self.number_of_states[section] + control]
        return div_back, div_front

    def states(self, state, section):
        """getter specify section states array
        Args:
            state (int) : state number
            section (int) : section number
        Returns:
            states ((N,) ndarray) :
                1-D array of state
        """
        div_back, div_front = self._division_states(state, section)
        return self.p[div_front:div_back] * self.unit_states[section][state]

    def states_all_section(self, state):
        """get states array
        Args:
            state (int) : state number
        Returns:
            states_all_section ((N,) ndarray) :
                1-D array of all section state
        """
        temp = np.zeros(0)
        for i in range(self.number_of_section):
            temp = np.concatenate([temp, self.states(state, i)])
        return temp

    def controls(self, control, section):
        """getter specify section controls array
        Args:
            control (int) : control number
            section (int) : section number
        Returns:
            controls (ndarray) :
                1-D array of controls
        """
        div_back, div_front = self._division_controls(control, section)
        return self.p[div_front:div_back] * self.unit_controls[section][control]

    def controls_all_section(self, control):
        """get controls array
        Args:
            control (int) : control number
        Returns:
            controls_all_section ((N, ) ndarray) :
                1-D array of all section control
        """
        temp = np.zeros(0)
        for i in range(self.number_of_section):
            temp = np.concatenate([temp, self.controls(control, i)])
        return temp

    def time_start(self, section):
        """ get time at section "start"
        Args:
            section (int) : section
        Returns:
            time_start (int) : time at section start
        """
        if (section == 0):
            return self.t0
        else:
            time_start_index = range(-self.number_of_section - 1, 0)
            return self.p[time_start_index[section]] * self.unit_time

    def time_final(self, section):
        """ get time at section "end"
        Args:
            section (int) : section
        Returns:
            time_final (int) : time at section end
        """
        time_final_index = range(-self.number_of_section, 0)
        return self.p[time_final_index[section]] * self.unit_time

    def time_final_all_section(self):
        """ get time at "end"
        Args:
            section (int) : section
        Returns:
            time_final_all_section (int) : time at end
        """
        tf = []
        for section in range(self.number_of_section):
            tf = tf + [self.time_final(section)]
        return tf

    def set_states(self, state, section, value):
        """set value to state at specific section
        Args:
            state (int) : state
            section (int) : section
            value (int) : value
        """
        assert len(value) == self.nodes[section], "Error: value length is NOT match nodes length"
        div_back, div_front = self._division_states(state, section)
        self.p[div_front:div_back] = value / self.unit_states[section][state]

    def set_states_all_section(self, state, value_all_section):
        """set value to state at all section
        Args:
            state (int) : state
            value_all_section (int) : value
        """
        div = 0
        for i in range(self.number_of_section):
            value = value_all_section[div:div + self.nodes[i]]
            div = div + self.nodes[i]
            self.set_states(state, i, value)

    def set_controls(self, control, section, value):
        """set value to control at all section
        Args:
            control (int) : control
            section (int) : section
            value (int) : value
        """
        assert len(value) == self.nodes[section], "Error: value length is NOT match nodes length"
        div_back, div_front = self._division_controls(control, section)
        self.p[div_front:div_back] = value / self.unit_controls[section][control]

    def set_controls_all_section(self, control, value_all_section):
        """set value to control at all section
        Args:
            control (int) : control
            value_all_section (int) : value
        """
        div = 0
        for i in range(self.number_of_section):
            value = value_all_section[div:div + self.nodes[i]]
            div = div + self.nodes[i]
            self.set_controls(control, i, value)

    def set_time_final(self, section, value):
        """ set value to final time at specific section
        Args:
            section (int) : seciton
            value (float) : value
        """
        time_final_index = range(-self.number_of_section, 0)
        self.p[time_final_index[section]] = value / self.unit_time

    def time_to_tau(self, time):
        time_init = min(time)
        time_final = max(time)
        time_center = (time_init + time_final) / 2
        temp = []
        for x in time:
            temp += [2 / (time_final - time_init) * (x - time_center)]
        return np.array(temp)

    def time_update(self):
        """ get time array after optimization
        Returns:
            time_update : (N,) ndarray
                time array
        """
        self.time = []
        t = [0] + self.time_final_all_section()
        for i in range(self.number_of_section):
            self.time.append((t[i+1] - t[i]) / 2.0 * self.tau[i]
                             + (t[i+1] + t[i]) / 2.0)
        return np.concatenate([i for i in self.time])

    def time_knots(self):
        """ get time at knot point
        Returns:
            time_knots (list) : time at knot point
        """
        return [0] + self.time_final_all_section()

    def index_states(self, state, section, index=None):
        """ get index of state at specific section
        Args:
            state (int) : state
            section (int) : section
            index (int, optional) : index
        Returns:
            index_states (int) : index of states
        """
        div_back, div_front = self._division_states(state, section)
        if (index is None):
            return div_front
        assert index < div_back - div_front, "Error, index out of range"
        if (index < 0):
            index = div_back - div_front + index
        return div_front + index

    def index_controls(self, control, section, index=None):
        div_back, div_front = self._division_controls(control, section)
        if (index is None):
            return div_front
        assert index < div_back - div_front, "Error, index out of range"
        if (index < 0):
            index = div_back - div_front + index
        return div_front + index

    def index_time_final(self, section):
        time_final_range = range(-self.number_of_section, 0)
        return self.number_of_variables + time_final_range[section]

    """
    ===========================
    UNIT SCALING ZONE
    ===========================
    """
    def set_unit_states(self, state, section, value):
        """ set a canonical unit value to the state at a specific section
        Args:
            state (int) : state
            section (int) : section
            value (float) : value
        """
        self.unit_states[section][state] = value

    def set_unit_states_all_section(self, state, value):
        """ set a canonical unit value to the state at all sections
        Args:
            state (int) : state
            value (float) : value
        """
        for i in range(self.number_of_section):
            self.set_unit_states(state, i, value)

    def set_unit_controls(self, control, section, value):
        """ set a canonical unit value to the control at a specific section
        Args:
            control (int) : control
            section (int) : section
            value (float) : value
        """
        self.unit_controls[section][control] = value

    def set_unit_controls_all_section(self, control, value):
        """ set a canonical unit value to the control at all sections
        Args:
            control (int) : control
            value (float) : value
        """
        for i in range(self.number_of_section):
            self.set_unit_controls(control, i, value)

    def set_unit_time(self, value):
        """ set a canonical unit value to the time
        Args:
            value (float) : value
        """
        self.unit_time = value
        time_init = np.array(self.time_init) / value
        self.time_init = list(time_init)
        self.time = []
        for index, node in enumerate(self.nodes):
            self.time.append((time_init[index+1] - time_init[index]) / 2.0 * self.tau[index]
                             + (time_init[index+1] + time_init[index]) / 2.0)
        self.t0 = time_init[0]
        self.time_all_section = np.concatenate([i for i in self.time])
        for section in range(self.number_of_section):
            self.set_time_final(section, time_init[section+1] * value)

    """ ==============================
    """

    def _dummy_func():
        pass

    """ ==============================
    """
    def solve(self, obj, display_func=_dummy_func, **options):
        """ solve NLP
        Args:
            obj (object instance) : instance
            display_func (function) : function to display intermediate values
            ftol (float, optional) : Precision goal for the value of f in the
                stopping criterion, (default: 1e-6)
            maxiter (int, optional) : Maximum number of iterations., (default : 25)
        Examples:
            "prob" is Problem class's instance.
            >>> prob.solve(obj, display_func, ftol=1e-12)
        """
        assert len(self.dynamics) != 0, "It must be set dynamics"
        assert self.cost is not None, "It must be set cost function"
        assert self.equality is not None, "It must be set equality function"
        assert self.inequality is not None, "It must be set inequality function"

        def equality_add(equality_func, obj):
            """ add pseudospectral method conditions to equality function.
            collocation point condition and knotting condition.
            """
            result = self.equality(self, obj)

            # collation point condition
            for i in range(self.number_of_section):
                D = self.D
                derivative = np.zeros(0)
                for j in range(self.number_of_states[i]):
                    state_temp = self.states(j, i) / self.unit_states[i][j]
                    derivative = np.hstack((derivative, D[i].dot(state_temp)))
                tix = self.time_start(i) / self.unit_time
                tfx = self.time_final(i) / self.unit_time
                dx = self.dynamics[i](self, obj, i)
                result = np.hstack((result, derivative - (tfx - tix) / 2.0 * dx))

            # knotting condition
            for knot in range(self.number_of_section - 1):
                if (self.number_of_states[knot] != self.number_of_states[knot + 1]):
                    continue  # if states are not continuous on knot, knotting condition skip
                for state in range(self.number_of_states[knot]):
                    param_prev = self.states(state, knot) / self.unit_states[knot][state]
                    param_post = self.states(state, knot + 1) / self.unit_states[knot][state]
                    if (self.knot_states_smooth[knot]):
                        result = np.hstack((result, param_prev[-1] - param_post[0]))

            return result

        def cost_add(cost_func, obj):
            """Combining nonintegrated function and integrated function.
            """
            not_integrated = self.cost(self, obj)
            if self.running_cost is None:
                return not_integrated
            integrand = self.running_cost(self, obj)
            weight = np.concatenate([i for i in self.w])
            integrated = sum(integrand * weight)
            return not_integrated + integrated

        def wrap_for_solver(func, arg0, arg1):
            def for_solver(p, arg0, arg1):
                self.p = p
                return func(arg0, arg1)
            return for_solver

        # def wrap_for_solver(func, *args):
        #     def for_solver(p, *args):
        #         self.p = p
        #         return func(*args)
        #     return for_solver

        cons = ({'type': 'eq',
                 'fun': wrap_for_solver(equality_add, self.equality, obj),
                'args': (self, obj,)},
                {'type': 'ineq',
                 'fun': wrap_for_solver(self.inequality, self, obj),
                 'args': (self, obj,)})

        if (self.cost_derivative is None):
            jac = None
        else:
            jac = wrap_for_solver(self.cost_derivative, self, obj)

        ftol = options.setdefault("ftol", 1e-6)
        maxiter = options.setdefault("maxiter", 25)
        #bnds = ((0.0, 1.0),) * len(self.p)
        #Vlb = np.zeros((len(self.p)))
        #Vub = np.ones((len(self.p)))
        #bnds = Bounds(Vlb, Vub)
        while self.iterator < self.maxIterator:
            print("---- iteration : {0} ----".format(self.iterator+1))
            opt = optimize.minimize(wrap_for_solver(cost_add, self.cost, obj),
                                    self.p,
                                    args=(self, obj),
                                    constraints=cons,
                                    jac=jac,
                                    method='SLSQP',
                                    options={"disp":True,
                                             "iprint":2,
                                             "maxiter": maxiter})
            print(opt.message)
            display_func()
            print("")
            if not(opt.status):
                break
            self.iterator += 1
    """ ==============================
    """

    def __init__(self, time_init, nodes, number_of_states, number_of_controls,
                 maxIterator = 100, method="LGL"):
        assert isinstance(time_init, list), \
            "error: time_init is not list"
        assert isinstance(nodes, list), \
            "error: nodes are not list"
        assert isinstance(number_of_states, list), \
            "error: number of states are not list"
        assert isinstance(number_of_controls, list), \
            "error: number of controls are not list"
        assert len(time_init) == len(nodes) + 1, \
            "error: time_init length is not match nodes length"
        assert len(nodes) == len(number_of_states), \
            "error: nodes length is not match states length"
        assert len(nodes) == len(number_of_controls), \
            "error: nodes length is not match controls length"
        self.nodes = nodes
        self.number_of_states = number_of_states
        self.number_of_controls = number_of_controls
        self.div = self._make_param_division(nodes, number_of_states, number_of_controls)
        self.number_of_section = len(self.nodes)
        self.number_of_param = np.array(number_of_states) + np.array(number_of_controls)
        self.number_of_variables = sum(self.number_of_param * nodes) + self.number_of_section
        self.tau = []
        self.w = []
        self.D = []
        self.time = []
        for index, node in enumerate(nodes):
            self.tau.append(self._nodes_LGL(node))
            self.w.append(self._weight_LGL(node))
            self.D.append(self._differentiation_matrix_LGL(node))
            self.time.append((time_init[index+1] - time_init[index]) / 2.0 * self.tau[index]
                             + (time_init[index+1] + time_init[index]) / 2.0)
        self.maxIterator = maxIterator
        self.iterator = 0
        self.time_init = time_init
        self.t0 = time_init[0]
        self.time_all_section = np.concatenate([i for i in self.time])
        # ====
        self.unit_states = []
        self.unit_controls = []
        self.unit_time = 1.0
        for i in range(self.number_of_section):
            self.unit_states.append([1.0]*self.number_of_states[i])
            self.unit_controls.append([1.0]*self.number_of_controls[i])
        # ====
        self.p = np.zeros(self.number_of_variables, dtype=float)
        # ====
        # function
        self.dynamics = []
        self.knot_states_smooth = []
        self.cost = None
        self.running_cost = None
        self.cost_derivative = None
        self.equality = None
        self.inequality = None
        # ====
        for section in range(self.number_of_section):
            self.set_time_final(section, time_init[section+1])
            self.dynamics.append(None)
        for section in range(self.number_of_section-1):
            self.knot_states_smooth.append(True)

    def __repr__(self):
        s = "---- parameter ----" + "\n"
        s += "nodes = " + str(self.nodes) + "\n"
        s += "number of states    = " + str(self.number_of_states) + "\n"
        s += "number of controls  = " + str(self.number_of_controls) + "\n"
        s += "number of sections  = " + str(self.number_of_section) + "\n"
        s += "number of variables = " + str(self.number_of_variables) + "\n"
        s += "---- algorithm ----" + "\n"
        s += "max iteration = " + str(self.maxIterator) + "\n"
        s += "---- function  ----" + "\n"
        s += "dynamics        = " + str(self.dynamics) + "\n"
        s += "cost            = " + str(self.cost) + "\n"
        s += "cost_derivative = " + str(self.cost_derivative) + "\n"
        s += "equality        = " + str(self.equality) + "\n"
        s += "inequality      = " + str(self.inequality) + "\n"
        s += "knot_states_smooth = " + str(self.dynamics) + "\n"

        return s

    def to_csv(self, filename="OpenGoddard_output.csv", delimiter=","):
        """ output states, controls and time to csv file
        Args:
            filename (str, optional) : csv filename
            delimiter : (str, optional) : default ","
        """
        result = np.zeros(0)
        result = np.hstack((result, self.time_update()))

        header = "time, "
        for i in range(self.number_of_states[0]):
            header += "state%d, " % (i)
            result = np.vstack((result, self.states_all_section(i)))
        for i in range(self.number_of_controls[0]):
            header += "control%d, " % (i)
            result = np.vstack((result, self.controls_all_section(i)))
        np.savetxt(filename, result.T, delimiter=delimiter, header=header)
        print("Completed saving \"%s\"" % (filename))

    def plot(self, title_comment=""):
        """ plot inner variables that to be optimized
        Args:
            title_comment (str) : string for title
        """
        plt.figure()
        plt.title("OpenGoddard inner variables" + title_comment)
        plt.plot(self.p, "o")
        plt.xlabel("variables")
        plt.ylabel("value")
        for section in range(self.number_of_section):
            for line in self.div[section]:
                plt.axvline(line, color="C%d" % ((section+1) % 6), alpha=0.5)
        plt.grid()


def fileReadOr(filename):
    '''function to read data from txt file'''
    with open(filename) as f:
        table = []
        for line in f:
            line = line.split()
            if line:
                line = [float(i) for i in line]
                table.append(line)

    table = table[1:][:]
    par = [[x[1], x[2], x[3], x[4], x[5], x[6]] for x in table]
    f.close()
    return par


class Spaceplane():

    def __init__(self):
        self.GMe = 3.986004418e14  # Earth gravitational constant [m^3/s^2]
        self.Re = 6371000  # Earth Radius [m]
        self.psl = 101325  # ambient pressure at sea level [Pa]
        self.latstart = np.deg2rad(5.2)  # deg latitude
        self.longstart = np.deg2rad(-52.775)  # deg longitude
        self.incl = np.deg2rad(51.6)  # deg orbit inclination
        self.gammastart = np.deg2rad(89)  # deg
        self.M0 = 450400  # kg  starting mass
        self.g0 = 9.80665  # m/s2
        self.gIsp = self.g0 * 455  # g0 * Isp max
        self.omega = 7.2921159e-5
        self.MaxQ = 40000  # Pa
        self.MaxAx = 30  # m/s2
        self.MaxAz = 15  # m/s2
        self.Htarget = 400000  # m target height after hohmann transfer
        self.wingSurf = 500.0  # m2
        self.lRef = 34.0  # m
        #self.k = 5e4  # [Nm] livello di precisione per trimmaggio
        self.m10 = self.M0 * 0.1
        self.xcgf = 0.37  # cg position with empty vehicle
        self.xcg0 = 0.65  # cg position at take-off
        self.pref = 21.25
        self.Hini = 100000
        self.r2 = self.Re + self.Htarget
        self.Rtarget = self.Re + self.Hini  # m/s
        self.Vtarget = np.sqrt(self.GMe / self.Rtarget)  # m/s forse da modificare con velocita' assoluta
        self.chi_fin = 0.5 * np.pi + np.arcsin(np.cos(self.incl) / np.cos(self.latstart))  # value in radians
        self.chistart = np.deg2rad(125)  # deg flight direction
        self.mach = np.array([0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0])
        self.angAttack = np.array([-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.5, 25.0, 30.0, 35.0, 40.0])
        #self.angAttack = np.linspace(-2.0, 40.0, 5)
        self.bodyFlap = np.array([-20, -10, 0, 10, 20, 30])
        #self.bodyFlap = np.linspace(-20, 30, 5)
        self.a = [-0.0065, 0, 0.0010, 0.0028, 0, -0.0020, -0.0040, 0]
        self.a90 = [0.0030, 0.0050, 0.0100, 0.0200, 0.0150, 0.0100, 0.0070]
        self.hv = [11000, 20000, 32000, 47000, 52000, 61000, 79000, 90000]
        self.h90 = [90000, 100000, 110000, 120000, 150000, 160000, 170000, 190000]
        self.tmcoeff = [180.65, 210.65, 260.65, 360.65, 960.65, 1110.65, 1210.65]
        self.pcoeff = [0.16439, 0.030072, 0.0073526, 0.0025207, 0.505861E-3, 0.36918E-3, 0.27906E-3]
        self.tcoeff2 = [2.937, 4.698, 9.249, 18.11, 12.941, 8.12, 5.1]
        self.tcoeff1 = [180.65, 210.02, 257.0, 349.49, 892.79, 1022.2, 1103.4]
        self.hvert = 100
        self.n = prob.nodes
        self.gammamax = np.deg2rad(89.9)
        self.gammamin = np.deg2rad(-50)
        self.chimax = np.deg2rad(150)
        self.chimin = np.deg2rad(100)
        self.lammax = np.deg2rad(30)
        self.lammin = np.deg2rad(2)
        self.tetamax = np.deg2rad(-10)
        self.tetamin = np.deg2rad(-70)
        self.hmax = 2e5
        self.hmin = 1.0
        self.vmax = 1e4
        self.vmin = 1.0
        self.alfamax = np.deg2rad(40)
        self.alfamin = np.deg2rad(-2)
        self.deltamax = 1.0
        self.deltamin = 0.0
        self.vstart = 0.1
        self.hstart = 0.1
        '''self.n10 = int(n[0] / 100 * 10)
        self.n90 = n[0] - self.n10
        self.n15 = int(n[0] / 100 * 15)
        self.n85 = n[0] - self.n15
        self.n20 = int(n[0] / 100 * 20)
        self.n80 = n[0] - self.n20'''
        self.n32 = int(n[0] / 100 * 20)
        #self.n70 = n[0] - self.n30
        self.n35 = int(n[0] / 100 * 35)
        #self.n65 = n[0] - self.n35
        self.n40 = int(n[0] / 100 * 40)
        self.n60 = n[0] - self.n40
        self.n45 = int(n[0] / 100 * 45)
        self.n55 = n[0] - self.n45
        self.n50 = int(n[0] / 100 * 50)
        '''self.alfa_lb = np.hstack((np.repeat(to_new_int(np.deg2rad(-1), np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0), self.n15),
                             np.repeat(to_new_int(np.deg2rad(-2), np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0), self.n85)))

        self.alfa_ub = np.hstack((np.repeat(to_new_int(np.deg2rad(1), np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0), self.n15),
                                  np.repeat(to_new_int(np.deg2rad(10), np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0),self.n15),
                             np.repeat(to_new_int(np.deg2rad(40), np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0), n[0] - 3*self.n15),
                                  np.repeat(to_new_int(np.deg2rad(5), np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0),self.n15)))

        self.delta_lb = np.hstack((np.repeat(0.99, self.n40), np.repeat(0.05, self.n60)))

        self.deltaf_lb = np.hstack((np.repeat(to_new_int(np.deg2rad(-1.0), np.deg2rad(-20), np.deg2rad(30), 0.0, 1.0), self.n15),
                                 np.repeat(to_new_int(np.deg2rad(-20), np.deg2rad(-20), np.deg2rad(30), 0.0, 1.0), n[0]-self.n15)))

        self.deltaf_ub = np.hstack((np.repeat(to_new_int(np.deg2rad(1.0), np.deg2rad(-20), np.deg2rad(30), 0.0, 1.0), self.n15),
                               np.repeat(to_new_int(np.deg2rad(30), np.deg2rad(-20), np.deg2rad(30), 0.0, 1.0), n[0] - self.n15)))

        self.mu_lb = np.hstack((np.repeat(to_new_int(np.deg2rad(-1.0), np.deg2rad(-80), np.deg2rad(80), 0.0, 1.0), self.n15),
                           np.repeat(to_new_int(np.deg2rad(-80), np.deg2rad(-80), np.deg2rad(80), 0.0, 1.0),
                                     n[0] - 2 * self.n15),
                           np.repeat(to_new_int(np.deg2rad(-1.0), np.deg2rad(-80), np.deg2rad(80), 0.0, 1.0), self.n15)))

        self.mu_ub = np.hstack((np.repeat(to_new_int(np.deg2rad(1.0), np.deg2rad(-80), np.deg2rad(80), 0.0, 1.0), self.n15),
                           np.repeat(to_new_int(np.deg2rad(60), np.deg2rad(-60), np.deg2rad(80), 0.0, 1.0),
                                     n[0] - 2 * self.n15),
                           np.repeat(to_new_int(np.deg2rad(1.0), np.deg2rad(-80), np.deg2rad(80), 0.0, 1.0), self.n15)))

        self.gamma_lb = np.hstack((np.repeat(to_new_int(np.deg2rad(65), np.deg2rad(-89.9), np.deg2rad(89.9), 0.0, 1.0), self.n35),
                              np.repeat(to_new_int(np.deg2rad(-89.9), np.deg2rad(-89.9), np.deg2rad(89.9), 0.0, 1.0),self.n65)))

        self.chi_ub = np.hstack((np.repeat(to_new_int(np.deg2rad(130), np.deg2rad(110), np.deg2rad(150), 0.0, 1.0), self.n10),
                            np.repeat(to_new_int(np.deg2rad(150), np.deg2rad(110), np.deg2rad(150), 0.0, 1.0), self.n90)))'''

    @staticmethod
    def isa(altitude, pstart, g0, r, flag):
        t0 = 288.15
        p0 = pstart
        prevh = 0.0
        R = 287.00
        m0 = 28.9644
        Rs = 8314.32
        m0 = 28.9644
        if flag == 1:
            altitude = np.array([altitude])
        temperature = np.zeros(len(altitude))
        pressure = np.zeros(len(altitude))
        tempm = np.zeros(len(altitude))
        density = np.zeros(len(altitude))
        csound = np.zeros(len(altitude))
        k = 0

        def cal(ps, ts, av, h0, h1):
            if av != 0:
                t1 = ts + av * (h1 - h0)
                p1 = ps * (t1 / ts) ** (-g0 / av / R)
            else:
                t1 = ts
                p1 = ps * np.exp(-g0 / R / ts * (h1 - h0))
            return t1, p1

        def atm90(a90v, z, hi, tc1, pc, tc2, tmc):
            for num in hi:
                if z <= num:
                    ind = hi.index(num)
                    if ind == 0:
                        zb = hi[0]
                        b = zb - tc1[0] / a90v[0]
                        t = tc1[0] + tc2[0] * (z - zb) / 1000
                        tm = tmc[0] + a90v[0] * (z - zb) / 1000
                        add1 = 1 / ((r + b) * (r + z)) + (1 / ((r + b) ** 2)) * np.log(abs((z - b) / (z + r)))
                        add2 = 1 / ((r + b) * (r + zb)) + (1 / ((r + b) ** 2)) * np.log(abs((zb - b) / (zb + r)))
                        p = pc[0] * np.exp(-m0 / (a90v[0] * Rs) * g0 * r ** 2 * (add1 - add2))
                    else:
                        zb = hi[ind - 1]
                        b = zb - tc1[ind - 1] / a90v[ind - 1]
                        t = tc1[ind - 1] + (tc2[ind - 1] * (z - zb)) / 1000
                        tm = tmc[ind - 1] + a90v[ind - 1] * (z - zb) / 1000
                        add1 = 1 / ((r + b) * (r + z)) + (1 / ((r + b) ** 2)) * np.log(abs((z - b) / (z + r)))
                        add2 = 1 / ((r + b) * (r + zb)) + (1 / ((r + b) ** 2)) * np.log(abs((zb - b) / (zb + r)))
                        p = pc[ind - 1] * np.exp(-m0 / (a90v[ind - 1] * Rs) * g0 * r ** 2 * (add1 - add2))
                    break
            return t, p, tm

        for alt in altitude:
            if alt < 0:
                # print("h < 0", alt)
                t = t0
                p = p0
                d = p / (R * t)
                c = np.sqrt(1.4 * R * t)
                density[k] = d
                csound[k] = c
                # temperature[k] = t
                pressure[k] = p
                # tempm[k] = t
            elif 0 <= alt < 90000:

                for i in range(0, 8):

                    if alt <= obj.hv[i]:
                        t, p = cal(p0, t0, obj.a[i], prevh, alt)
                        d = p / (R * t)
                        c = np.sqrt(1.4 * R * t)
                        density[k] = d
                        csound[k] = c
                        temperature[k] = t
                        pressure[k] = p
                        tempm[k] = t
                        t0 = 288.15
                        p0 = pstart
                        prevh = 0
                        break
                    else:

                        t0, p0 = cal(p0, t0, obj.a[i], prevh, obj.hv[i])
                        prevh = obj.hv[i]

            elif 90000 <= alt <= 190000:
                t, p, tpm = atm90(obj.a90, alt, obj.h90, obj.tcoeff1, obj.pcoeff, obj.tcoeff2, obj.tmcoeff)
                d = p / (R * tpm)
                c = np.sqrt(1.4 * R * tpm)
                density[k] = d
                csound[k] = c
                # temperature[k] = t
                pressure[k] = p
                # tempm[k] = t
            elif alt > 190000:
                # print("h > 190 km", alt)
                zb = obj.h90[6]
                z = obj.h90[-1]
                b = zb - obj.tcoeff1[6] / obj.a90[6]
                t = obj.tcoeff1[6] + (obj.tcoeff2[6] * (z - zb)) / 1000
                tm = obj.tmcoeff[6] + obj.a90[6] * (z - zb) / 1000
                add1 = 1 / ((r + b) * (r + z)) + (1 / ((r + b) ** 2)) * np.log(abs((z - b) / (z + r)))
                add2 = 1 / ((r + b) * (r + zb)) + (1 / ((r + b) ** 2)) * np.log(abs((zb - b) / (zb + r)))
                p = obj.pcoeff[6] * np.exp(-m0 / (obj.a90[6] * Rs) * g0 * r ** 2 * (add1 - add2))
                d = p / (R * t)
                c = np.sqrt(1.4 * R * tm)
                density[k] = d
                csound[k] = c
                # temperature[k] = t
                pressure[k] = p
                # tempm[k] = t
            k += 1
        return pressure, density, csound

    @staticmethod
    def aeroForces(M, alfa, deltaf, cd, cl, cm, v, sup, rho, leng, mstart, mass, m10, xcg0, xcgf, pref, npoint):
        alfag = np.rad2deg(alfa)
        deltafg = np.rad2deg(deltaf)
        coeffs = np.array((cl, cd, cm))
        if npoint == 1:
            M = np.array([M])
            alfag = np.array([alfag])
            deltafg = np.array([deltafg])
        cL, cD, cM1 = pool.map(partial(c_eval, npoint, M, alfag, deltafg, coeffs, obj.mach, obj.angAttack, obj.bodyFlap), range(3))

        L = 0.5 * (v ** 2) * sup * rho * cL
        D = 0.5 * (v ** 2) * sup * rho * cD
        xcg = leng * (((xcgf - xcg0) / (m10 - mstart)) * (mass - mstart) + xcg0)
        Dx = xcg - pref
        cM = cM1 + cL * (Dx / leng) * np.cos(alfa) + cD * (Dx / leng) * np.sin(alfa)
        Mom = 0.5 * (v ** 2) * sup * leng * rho * cM

        return L, D, Mom

    @staticmethod
    def thrust(presamb, mass, presv, spimpv, delta, tau, npoint, slpres, wlo, we, lref, xcgf, xcg0):
        nimp = 17
        nmot = 1
        Thrust = np.zeros(npoint)
        Deps = np.zeros(npoint)
        Simp = np.zeros(npoint)
        Mom = np.zeros(npoint)
        if npoint == 1:
            delta = np.array([delta])
            tau = np.array([tau])
            mass = np.array([mass])
        for j in range(npoint):
            thrx = nmot * (5.8e6 + 14.89 * slpres - 11.16 * presamb[j]) * delta[j]
            if presamb[j] >= slpres:
                presamb[j] = slpres
                spimp = spimpv[-1]
            elif presamb[j] < slpres:
                for i in range(nimp):
                    if presv[i] >= presamb[j]:
                        spimp = splev(presamb[j], spimp_interp, der=0)
                        break

            xcg = ((xcgf - xcg0) / (we - wlo) * (mass[j] - wlo) + xcg0) * lref
            dthr = 0.4224 * (36.656 - xcg) * thrx - 19.8 * (32 - xcg) * (1.7 * slpres - presamb[j])
            mommot = tau[j] * dthr
            thrz = -tau[j] * (2.5e6 - 22 * slpres + 9.92 * presamb[j])
            thrust = np.sqrt(thrx ** 2 + thrz ** 2)

            if thrx == 0.0 and thrz == 0.0:
                deps = 0.0
            else:
                deps = np.arctan(thrz / thrx)

            Thrust[j] = thrust
            Deps[j] = deps
            Simp[j] = spimp
            Mom[j] = mommot
        return Thrust, Deps, Simp, Mom


def c_eval(n, m, a, b, cc, mach, angAttack, bodyFlap, it):
    coeff = cc[it]
    c = np.zeros(n)
    for i in range(n):
        c[i] = coefCalc(coeff, m[i], a[i], b[i], mach, angAttack, bodyFlap)
    return c


def limCalc(array, value):
    j = 0
    lim = array.__len__()
    for num in array:
        if j == lim - 1:
            sup = num
            inf = array[j - 1]
        if value < num:
            sup = num
            if j == 0:
                inf = num
            else:
                inf = array[j - 1]
            break
        j += 1
    s = np.where(array == sup)[0][0]
    i = np.where(array == inf)[0][0]
    return i, s


def coefCalc(coeff, m, alfa, deltaf, mach, angAttack, bodyFlap):
    if m > mach[-1]:
        m = mach[-1]
    elif m < mach[0]:
        m = mach[0]
    if alfa > angAttack[-1]:
        alfa = angAttack[-1]
    elif alfa < angAttack[0]:
        alfa = angAttack[0]
    if deltaf > bodyFlap[-1]:
        deltaf = bodyFlap[-1]
    elif deltaf < bodyFlap[0]:
        deltaf = bodyFlap[0]

    (im, sm), (ia, sa), (idf, sdf) = map(limCalc, (mach, angAttack, bodyFlap), (m, alfa, deltaf))
    cnew1 = coeff[im]  # [17 * im: 17 * im + angAttack.__len__()][:]
    cnew2 = coeff[sm]  # [17 * sm: 17 * sm + angAttack.__len__()][:]

    '''interpolation on the first table between angle of attack and deflection'''
    rowinf1 = cnew1[ia][:]
    rowsup1 = cnew1[sa][:]
    coeffinf = [rowinf1[idf], rowsup1[idf]]
    coeffsup = [rowinf1[sdf], rowsup1[sdf]]
    c1 = coeffinf[0] + (alfa - angAttack[ia]) * (
            (coeffinf[1] - coeffinf[0]) / (angAttack[sa] - angAttack[ia]))
    c2 = coeffsup[0] + (alfa - angAttack[ia]) * (
            (coeffsup[1] - coeffsup[0]) / (angAttack[sa] - angAttack[ia]))
    coeffd1 = c1 + (deltaf - bodyFlap[idf]) * ((c2 - c1) / (bodyFlap[sdf] - bodyFlap[idf]))
    '''interpolation on the first table between angle of attack and deflection'''
    rowinf2b = cnew2[ia][:]
    rowsup2b = cnew2[sa][:]
    coeffinfb = [rowinf2b[idf], rowsup2b[idf]]
    coeffsupb = [rowinf2b[sdf], rowsup2b[sdf]]
    c1b = coeffinfb[0] + (alfa - angAttack[ia]) * (
            (coeffinfb[1] - coeffinfb[0]) / (angAttack[sa] - angAttack[ia]))
    c2b = coeffsupb[0] + (alfa - angAttack[ia]) * (
            (coeffsupb[1] - coeffsupb[0]) / (angAttack[sa] - angAttack[ia]))
    coeffd2 = c1b + (deltaf - bodyFlap[idf]) * ((c2b - c1b) / (bodyFlap[sdf] - bodyFlap[idf]))
    '''interpolation on the moments to obtain final coefficient'''
    coeffFinal = coeffd1 + (m - mach[im]) * ((coeffd2 - coeffd1) / (mach[sm] - mach[im]))

    return coeffFinal


def to_new_int(t, a, b, c, d):
    ''' this function converts a value from an interval [a, b] to [c, d]
    t = value to be converted
    a = inf lim old interval
    b = sup lim old interval
    c = inf lim new interval
    d = sup lim new interval'''
    return c + ((d - c) / (b - a)) * (t - a)


def dynamics(prob, obj, section):
    v = prob.states(0, section)
    chi = prob.states(1, section)
    gamma = prob.states(2, section)
    # teta = prob.states(3, section)
    lam = prob.states(4, section)
    h = prob.states(5, section)
    m = prob.states(6, section)
    #time = prob.time_update()
    alfa = prob.controls(0, section)
    #part1 = np.repeat(1.0, int(len(v) * 0.4))
    #part2 = Guess.linear(time[int(len(v) * 0.4):], 1.0, 0.0001)
    #delta = np.hstack((part1, part2))
    delta = prob.controls(1, section)
    deltaf = np.zeros(len(v)) #prob.controls(2, section)
    tau = np.zeros(len(v)) #prob.controls(2, section)
    mu = np.zeros(len(v)) #prob.controls(4, section)

    Press, rho, c = obj.isa(h, obj.psl, obj.g0, obj.Re, 0)

    M = v / c

    L, D, MomA = obj.aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                                obj.xcg0, obj.xcgf, obj.pref, prob.nodes[0])

    T, Deps, isp, MomT = obj.thrust(Press, m, presv, spimpv, delta, tau, prob.nodes[0], obj.psl, obj.M0, obj.m10,
                                    obj.lRef, obj.xcgf, obj.xcg0)

    eps = Deps + alfa
    g0 = obj.g0
    g = np.zeros(prob.nodes[0])
    k = 0
    for alt in h:
        if alt == 0:
            g[k] = g0
            k += 1
        else:
            g[k] = obj.g0 * (obj.Re / (obj.Re + alt)) ** 2  # [m/s2]
            k += 1
    hi = 0
    for i in range(len(h)):
        if h[i] > obj.hvert:
            hi = i-1
            break
    if hi <= 0:
        hi = 1

    dx = Dynamics(prob, section)

    dx[0] = (T[0:hi]*np.cos(eps[0:hi]) - D[0:hi])/m[0:hi] - g[0:hi]
    dx[1] = np.zeros((hi))
    dx[2] = np.zeros((hi))
    dx[3] = np.zeros((hi))
    dx[4] = np.zeros((hi))
    dx[5] = v[0:hi]
    dx[6] = -T[0:hi] / (g0 * isp[0:hi])


    dx[0] = np.hstack((dx[0], (T[hi:] * np.cos(eps[hi:]) - D[hi:]) / m[hi:] - g[hi:] * np.sin(gamma[hi:]) + (obj.omega ** 2) * (obj.Re + h[hi:]) * np.cos(lam[hi:]) * \
            (np.cos(lam[hi:]) * np.sin(gamma[hi:]) - np.sin(lam[hi:]) * np.cos(gamma[hi:]) * np.sin(chi[hi:]))))
    dx[1] = np.hstack((dx[1], ((T[hi:] * np.sin(eps[hi:]) + L[hi:]) * np.sin(mu[hi:])) / (m[hi:] * v[hi:] * np.cos(gamma[hi:])) - np.cos(gamma[hi:]) * np.cos(chi[hi:]) * np.tan(lam[hi:]) \
            * (v[hi:] / (obj.Re + h[hi:])) + 2 * obj.omega * (np.cos(lam[hi:]) * np.tan(gamma[hi:]) * np.sin(chi[hi:]) - np.sin(lam[hi:])) \
            - (obj.omega ** 2) * ((obj.Re + h[hi:]) / (v[hi:] * np.cos(gamma[hi:]))) * np.cos(lam[hi:]) * np.sin(lam[hi:]) * np.cos(chi[hi:])))
    dx[2] = np.hstack((dx[2], ((T[hi:] * np.sin(eps[hi:]) + L[hi:]) * np.cos(mu[hi:])) / (m[hi:] * v[hi:]) - (g[hi:] / v[hi:] - v[hi:] / (obj.Re + h[hi:])) * np.cos(gamma[hi:]) + 2 * obj.omega \
            * np.cos(lam[hi:]) * np.cos(chi[hi:]) + (obj.omega ** 2) * ((obj.Re + h[hi:]) / v[hi:]) * np.cos(lam[hi:]) * \
            (np.sin(lam[hi:]) * np.sin(gamma[hi:]) * np.sin(chi[hi:]) + np.cos(lam[hi:]) * np.cos(gamma[hi:]))))
    dx[3] = np.hstack((dx[3], -np.cos(gamma[hi:]) * np.cos(chi[hi:]) * (v[hi:] / ((obj.Re + h[hi:]) * np.cos(lam[hi:])))))
    dx[4] = np.hstack((dx[4], np.cos(gamma[hi:]) * np.sin(chi[hi:]) * (v[hi:] / (obj.Re + h[hi:]))))
    dx[5] = np.hstack((dx[5], v[hi:] * np.sin(gamma[hi:])))
    dx[6] = np.hstack((dx[6], -T[hi:] / (g0 * isp[hi:])))
    return dx()


def cost(prob, obj):
    h = prob.states_all_section(5)
    m = prob.states_all_section(6)
    #time = prob.time_update()
    #part1 = np.repeat(1.0, int(len(m) * 0.4))
    #part2 = Guess.linear(time[int(len(m) * 0.4):], 1.0, 0.0001)
    #delta = np.hstack((part1, part2))
    delta = prob.controls_all_section(1)
    tau = np.zeros(len(h)) #prob.controls_all_section(2)

    Press, rho, c = obj.isa(h, obj.psl, obj.g0, obj.Re, 0)

    T, Deps, isp, MomT = obj.thrust(Press, m, presv, spimpv, delta, tau, prob.nodes[0], obj.psl, obj.M0, obj.m10,
                                    obj.lRef, obj.xcgf, obj.xcg0)

    r1 = h[-1] + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m[-1] / (np.exp(Dv1 / obj.gIsp) * np.exp(Dv2 / (obj.g0 * isp[-1])))

    return -mf / obj.M0


def equality(prob, obj):
    v = prob.states_all_section(0)
    chi = prob.states_all_section(1)
    gamma = prob.states_all_section(2)
    teta = prob.states_all_section(3)
    lam = prob.states_all_section(4)
    h = prob.states_all_section(5)
    m = prob.states_all_section(6)

    alfa = prob.controls_all_section(0)
    #part1 = np.repeat(1.0, int(len(v) * 0.4))
    #time = prob.time_update()
    #part2 = Guess.linear(time[int(len(v) * 0.4):], 1.0, 0.0001)
    #delta = np.hstack((part1, part2))
    delta = prob.controls_all_section(1)
    #deltaf = np.zeros(len(v)) #prob.controls_all_section(2)
    #tau = prob.controls_all_section(2)
    #mu = np.zeros(len(v)) #prob.controls_all_section(4)

    States = np.array((v[-1], chi[-1], gamma[-1], teta[-1], lam[-1], h[-1], m[-1]))
    Controls = np.array((alfa[-1], delta[-1])) #deltaf[-1], tau[-1], mu[-1]))
    vt = np.sqrt(obj.GMe / (obj.Re + h[-1]))  # - obj.omega*np.cos(lam[-1])*(obj.Re+h[-1])

    def vass(states, controls, dyn, omega, obj):
        Re = 6371000
        v = states[0]
        chi = states[1]
        gamma = states[2]
        teta = states[3]
        lam = states[4]
        h = states[5]

        vv = np.array((-v * np.cos(gamma) * np.cos(chi),
                       v * np.cos(gamma) * np.sin(chi),
                       -v * np.sin(gamma)))
        vv[0] = vv[0] + omega * np.cos(lam) * (Re + h)
        vela2 = np.sqrt(vv[0] ** 2 + vv[1] ** 2 + vv[2] ** 2)
        if vv[0] <= 0.0:
            if abs(vv[0]) >= abs(vv[1]):
                chiass = np.arctan(abs(vv[1] / vv[0]))
                if vv[1] < 0.0:
                    chiass = -chiass
            elif abs(vv[0]) < abs(vv[1]):
                chiass = np.pi * 0.5 - np.arctan(abs(vv[0] / vv[1]))
                if vv[1] < 0.0:
                    chiass = -chiass
        elif vv[0] > 0.0:
            if abs(vv[0]) >= abs(vv[1]):
                chiass = np.pi - np.arctan((abs(vv[1] / vv[0])))
                if vv[1] < 0.0:
                    chiass = - chiass
            elif abs(vv[0]) < abs(vv[1]):
                chiass = np.pi * 0.5 + np.arctan(abs(vv[0] / vv[1]))
                if vv[1] < 0.0:
                    chiass = -chiass

        x = np.array(((Re + h) * np.cos(lam) * np.cos(teta),
                      (Re + h) * np.cos(lam) * np.sin(teta),
                      (Re + h) * np.sin(lam)))

        dx = dyn(states, controls, obj)
        xp = np.array(
            (dx[5] * np.cos(lam) * np.cos(teta) - (Re + h) * dx[4] * np.sin(lam) * np.cos(teta) - (Re + h) * dx[3]
             * np.cos(lam) * np.sin(teta),
             dx[5] * np.cos(lam) * np.sin(teta) - (Re + h) * dx[4] * np.sin(lam) * np.sin(teta) + (Re + h) * dx[3]
             * np.cos(lam) * np.cos(teta),
             dx[5] * np.sin(lam) + (Re + h) * dx[4] * np.cos(lam)))

        dxp = np.array((-omega * x[1],
                        omega * x[0],
                        0.0))

        vtot = np.array((xp[0] + dxp[0],
                         xp[1] + dxp[1],
                         xp[2] + dxp[2]))

        vela = np.sqrt(vtot[0] ** 2 + vtot[1] ** 2 + vtot[2] ** 2)

        return vela, chiass, vela2

    vtAbs, chiass, vtAbs2 = vass(States, Controls, dynamicsVel, obj.omega, obj)

    if np.cos(obj.incl) > np.cos(lam[-1]):
        chifin = np.pi
    else:
        chifin = 0.5 * np.pi + np.arcsin(np.cos(obj.incl) / np.cos(lam[-1]))
    result = Condition()

    # event condition
    result.equal(to_new_int(v[0], obj.vmin, obj.vmax, 0.0, 1.0), to_new_int(obj.vstart, obj.vmin, obj.vmax, 0.0, 1.0), unit=1)      #########   O L D  ##########
    #result.equal(to_new_int(chi[0], obj.chimin, obj.chimax, 0.0, 1.0), to_new_int(obj.chistart, obj.chimin, obj.chimax, 0.0, 1.0), unit=1)
    result.equal(to_new_int(gamma[0], obj.gammamin, obj.gammamax, 0.0, 1.0), to_new_int(obj.gammastart, obj.gammamin, obj.gammamax, 0.0, 1.0), unit=1)
    result.equal(to_new_int(teta[0], obj.tetamin, obj.tetamax, 0.0, 1.0), to_new_int(obj.longstart, obj.tetamin, obj.tetamax, 0.0, 1.0), unit=1)
    result.equal(to_new_int(lam[0], obj.lammin, obj.lammax, 0.0, 1.0), to_new_int(obj.latstart, obj.lammin, obj.lammax, 0.0, 1.0), unit=1)
    result.equal(to_new_int(h[0], obj.hmin, obj.hmax, 0.0, 1.0), to_new_int(obj.hstart, obj.hmin, obj.hmax, 0.0, 1.0), unit=1)
    result.equal(to_new_int(m[0], obj.m10, obj.M0, 0.0, 1.0), to_new_int(obj.M0, obj.m10, obj.M0, 0.0, 1.0), unit=1)
    # result.equal(to_new_int(alfa[0], np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0), to_new_int(0.0, np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0), unit=1)
    # result.equal(to_new_int(deltaf[0], np.deg2rad(-20), np.deg2rad(30), 0.0, 1.0), to_new_int(0.0, np.deg2rad(-20), np.deg2rad(30), 0.0, 1.0), unit=1)
    # result.equal(to_new_int(tau[0], -1, 1, 0.0, 1.0), to_new_int(0.0, -1, 1, 0.0, 1.0), unit=1)
    # result.equal(to_new_int(mu[0], np.deg2rad(-60), np.deg2rad(60), 0.0, 1.0), to_new_int(0.0, np.deg2rad(-60), np.deg2rad(60), 0.0, 1.0), unit=1)
    #result.equal(to_new_int(mu[-1], np.deg2rad(-60), np.deg2rad(60), 0.0, 1.0), to_new_int(0.0, np.deg2rad(-60), np.deg2rad(60), 0.0, 1.0), unit=1)
    result.equal(to_new_int(vtAbs, obj.vmin, obj.vmax, 0.0, 1.0), to_new_int(vt, obj.vmin, obj.vmax, 0.0, 1.0), unit=1)
    result.equal(to_new_int(chiass, obj.chimin, obj.chimax, 0.0, 1.0), to_new_int(chifin, obj.chimin, obj.chimax, 0.0, 1.0), unit=1)
    result.equal(to_new_int(gamma[-1], obj.gammamin, obj.gammamax, 0.0, 1.0), to_new_int(0.0, obj.gammamin, obj.gammamax, 0.0, 1.0), unit=1)

    '''result.equal(v[0], obj.vstart, unit=obj.vmax)
    result.equal(chi[0], obj.chistart, unit=obj.chimax)
    result.equal(gamma[0], obj.gammastart, unit=obj.gammastart)
    result.equal(teta[0], obj.longstart, unit=obj.tetamax)
    result.equal(lam[0], obj.latstart, unit=obj.lammax)
    result.equal(h[0], obj.hstart, unit=obj.hmax)
    result.equal(m[0], obj.M0, unit=obj.M0)
    result.equal(delta[0], 1.0, unit=obj.deltamax)
    result.equal(vtAbs, vt, unit=obj.vmax)
    result.equal(chiass, chifin, unit=obj.chimax)
    result.equal(gamma[-1], 0.0, unit=obj.gammamax)'''

    return result()


def inequality(prob, obj):
    v = prob.states_all_section(0)
    chi = prob.states_all_section(1)
    gamma = prob.states_all_section(2)
    teta = prob.states_all_section(3)
    lam = prob.states_all_section(4)
    h = prob.states_all_section(5)
    m = prob.states_all_section(6)
    time = prob.time_update()
    alfa = prob.controls_all_section(0)
    #part1 = np.repeat(1.0, int(len(v) * 0.4))
    #part2 = Guess.linear(time[int(len(v) * 0.4):], 1.0, 0.0001)
    #delta = np.hstack((part1, part2))
    delta = prob.controls_all_section(1)
    deltaf = np.zeros(len(v)) #prob.controls_all_section(2)
    tau = np.zeros(len(v)) #prob.controls_all_section(2)
    #mu = np.zeros(len(v)) #prob.controls_all_section(4)
    #t = prob.time_update()

    Press, rho, c = obj.isa(h, obj.psl, obj.g0, obj.Re, 0)

    M = v / c

    L, D, MomA = obj.aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                                obj.xcg0, obj.xcgf, obj.pref, prob.nodes[0])

    T, Deps, isp, MomT = obj.thrust(Press, m, presv, spimpv, delta, tau, prob.nodes[0], obj.psl, obj.M0, obj.m10,
                                    obj.lRef, obj.xcgf, obj.xcg0)

    #MomTot = MomA + MomT

    # dynamic pressure

    q = 0.5 * rho * (v ** 2)
    # accelerations
    ax = (T * np.cos(Deps) - D * np.cos(alfa) + L * np.sin(alfa)) / m
    az = (T * np.sin(Deps) + D * np.sin(alfa) + L * np.cos(alfa)) / m

    r1 = h[-1] + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m[-1] / (np.exp(Dv1 / obj.gIsp) * np.exp(Dv2 / (obj.g0 * isp[-1])))

    #diff_chi = np.diff(np.rad2deg(chi))
    #diff_gamma = np.diff(np.rad2deg(gamma))
    #diff_alfa = np.diff(np.rad2deg(alfa))
    #diff_delta = np.diff(delta * 100)
    #diff_deltaf = np.diff(np.rad2deg(deltaf))
    #diff_tau = np.diff(tau * 100)
    #diff_mu = np.diff(np.rad2deg(mu))
    #diff_t = np.diff(t)
    # print(diff_t)
    #mchi = abs(diff_chi / diff_t)
    #mgamma = abs(diff_gamma / diff_t)
    #malfa = abs(diff_alfa / diff_t)
    #mdelta = abs(diff_delta / diff_t)
    #mtau = abs(diff_tau / diff_t)
    #mdeltaf = abs(diff_deltaf / diff_t)
    #mmu = abs(diff_mu / diff_t)
    #mthrot = np.hstack((mdelta, mtau))
    result = Condition()

    # lower bounds
    result.lower_bound(to_new_int(v, obj.vmin, obj.vmax, 0.0, 1.0), to_new_int(obj.vmin, obj.vmin, obj.vmax, 0.0, 1.0), unit=1)  # v lower bound                                   ############### OLD  ###################
    result.lower_bound(to_new_int(chi, obj.chimin, obj.chimax, 0.0, 1.0), to_new_int(obj.chimin, obj.chimin, obj.chimax, 0.0, 1.0), unit=1)  # chi lower bound
    result.lower_bound(to_new_int(gamma, obj.gammamin, obj.gammamax, 0.0, 1.0), to_new_int(obj.gammamin, obj.gammamin, obj.gammamax, 0.0, 1.0), unit=1)  # gamma lower bound
    result.lower_bound(to_new_int(teta, obj.tetamin, obj.tetamax, 0.0, 1.0), to_new_int(obj.tetamin, obj.tetamin, obj.tetamax, 0.0, 1.0), unit=1)  # teta lower bound
    result.lower_bound(to_new_int(lam, obj.lammin, obj.lammax, 0.0, 1.0), to_new_int(obj.lammin, obj.lammin, obj.lammax, 0.0, 1.0), unit=1)  # lambda lower bound
    result.lower_bound(to_new_int(h, obj.hmin, obj.hmax, 0.0, 1.0), to_new_int(obj.hmin, obj.hmin, obj.hmax, 0.0, 1.0), unit=1)  # h lower bound
    result.lower_bound(to_new_int(m, obj.m10, obj.M0, 0.0, 1.0), to_new_int(obj.m10, obj.m10, obj.M0, 0.0, 1.0), unit=1)  # m lower bound
    result.lower_bound(to_new_int(alfa, np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0), to_new_int(np.deg2rad(-2), np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0), unit=1)  # alpha lower bound
    result.lower_bound(delta, obj.deltamin, unit=1)  # delta lower bound
    # result.lower_bound(to_new_int(deltaf, np.deg2rad(-20), np.deg2rad(30), 0.0, 1.0),  # obj.deltaf_lb, unit=1), to_new_int(np.deg2rad(-20), np.deg2rad(-20), np.deg2rad(30), 0.0, 1.0), unit=1)  # deltaf lower bound
    # result.lower_bound(to_new_int(tau, -1, 1, 0, 1), to_new_int(-1, -1, 1, 0, 1), unit=1)  # tau lower bound
    # result.lower_bound(to_new_int(mu, np.deg2rad(-60), np.deg2rad(60), 0.0, 1.0),  to_new_int(np.deg2rad(-60), np.deg2rad(-60), np.deg2rad(60), 0.0, 1.0), unit=1)  # mu lower bound
    result.lower_bound(to_new_int(mf, obj.m10, obj.M0, 0.0, 1.0), to_new_int(obj.m10, obj.m10, obj.M0, 0.0, 1.0), unit=1)  # mf lower bound
    result.lower_bound(to_new_int(h[-1], obj.hmin, obj.hmax, 0.0, 1.0), to_new_int(6e4, obj.hmin, obj.hmax, 0.0, 1.0), unit=1)  # final h lower bound
    # result.lower_bound(to_new_int(MomTot / 1e4, -1e2, 1e2, 0.0, 1.0), to_new_int(-obj.k / 1e4, -1e2, 1e2, 0.0, 1.0), unit=1)  # total moment lower bound
    result.lower_bound(to_new_int(az, -1e2, 1e2, 0.0, 1.0), to_new_int(-obj.MaxAz, -1e2, 1e2, 0.0, 1.0), unit=1)  # ax lower bound
    result.lower_bound(to_new_int(v[-1], obj.vmin, obj.vmax, 0.0, 1.0), to_new_int(6e3, obj.vmin, obj.vmax, 0.0, 1.0),
                       unit=1)  # v lower bound                                   ############### OLD  ###################

    '''result.lower_bound(v, obj.vmin, unit=obj.vmax)  # v lower bound
    result.lower_bound(chi, obj.chimin, unit=obj.chimax)  # chi lower bound
    result.lower_bound(gamma, obj.gammamin, unit=obj.gammamax)  # gamma lower bound
    result.lower_bound(teta, obj.tetamin, unit=obj.tetamax)  # teta lower bound
    result.lower_bound(lam, obj.lammin, unit=obj.lammax)  # lambda lower bound
    result.lower_bound(h, obj.hmin, unit=obj.hmax)  # h lower bound
    result.lower_bound(m, obj.m10, unit=obj.M0)  # m lower bound
    result.lower_bound(alfa, obj.alfamin, unit=obj.alfamax)  # alpha lower bound
    result.lower_bound(delta, obj.deltamin, unit=obj.deltamax)  # delta lower bound
    result.lower_bound(mf, obj.m10, unit=obj.M0)  # mf lower bound
    result.lower_bound(h[-1], 6e4, unit=obj.hmax)  # final h lower bound
    result.lower_bound(az, -obj.MaxAz, unit=obj.MaxAz)  # ax lower bound'''

    # upper bounds
    result.upper_bound(to_new_int(v, obj.vmin, obj.vmax, 0.0, 1.0), to_new_int(obj.vmax, obj.vmin, obj.vmax, 0.0, 1.0),
                       unit=1)  # v upper bound

    result.upper_bound(to_new_int(chi, obj.chimin, obj.chimax, 0.0, 1.0),  # obj.chi_ub, unit=1)
                       to_new_int(obj.chimax, obj.chimin, obj.chimax, 0.0, 1.0),
                       unit=1)  # chi upper bound

    result.upper_bound(to_new_int(gamma, obj.gammamin, obj.gammamax, 0.0, 1.0),
                       to_new_int(obj.gammamax, obj.gammamin, obj.gammamax, 0.0, 1.0),
                       unit=1)  # gamma upper bound

    result.upper_bound(to_new_int(teta, obj.tetamin, obj.tetamax, 0.0, 1.0),
                       to_new_int(obj.tetamax, obj.tetamin, obj.tetamax, 0.0, 1.0), unit=1)  # teta upper bound

    result.upper_bound(to_new_int(lam, obj.lammin, obj.lammax, 0.0, 1.0),
                       to_new_int(obj.lammax, obj.lammin, obj.lammax, 0.0, 1.0), unit=1)  # lam upper bound

    result.upper_bound(to_new_int(h, obj.hmin, obj.hmax, 0.0, 1.0), to_new_int(obj.hmax, obj.hmin, obj.hmax, 0.0, 1.0),
                       unit=1)  # h upper bound

    result.upper_bound(to_new_int(m, obj.m10, obj.M0, 0.0, 1.0), to_new_int(obj.M0, obj.m10, obj.M0, 0.0, 1.0),
                       unit=1)  # m upper bound

    result.upper_bound(to_new_int(alfa, np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0),  # obj.alfa_ub, unit=1)
                       to_new_int(np.deg2rad(40), np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0), unit=1)  # alfa upper bound

    result.upper_bound(delta, obj.deltamax, unit=1)  # delta upper bound

    #result.upper_bound(to_new_int(deltaf, np.deg2rad(-20), np.deg2rad(30), 0.0, 1.0),  # obj.deltaf_ub, unit=1)
     #                  to_new_int(np.deg2rad(30), np.deg2rad(-20), np.deg2rad(30), 0.0, 1.0),
      #                 unit=1)  # deltaf upper bound

    #result.upper_bound(to_new_int(tau, -1, 1, 0, 1), to_new_int(1, -1, 1, 0, 1), unit=1)  # tau upper bound

    #result.upper_bound(to_new_int(mu, np.deg2rad(-60), np.deg2rad(60), 0.0, 1.0),  # obj.mu_ub, unit=1)
     #                  to_new_int(np.deg2rad(60), np.deg2rad(-60), np.deg2rad(60), 0.0, 1.0), unit=1)  # mu upper bound

    #result.upper_bound(to_new_int(MomTot / 1e5, -1e2, 1e2, 0.0, 1.0), to_new_int(obj.k / 1e5, -1e2, 1e2, 0.0, 1.0),
     #                  unit=1)  # momtot upper bound

    result.upper_bound(to_new_int(az, -1e2, 1e2, 0.0, 1.0), to_new_int(obj.MaxAz, -1e2, 1e2, 0.0, 1.0),
                       unit=1)  # az upper bound

    result.upper_bound(to_new_int(ax, 0.0, 1e2, 0.0, 1.0), to_new_int(obj.MaxAx, 0.0, 1e2, 0.0, 1.0),
                       unit=1)  # ax upper bound

    result.upper_bound(to_new_int(q, 0.0, 5e4, 0.0, 1.0), to_new_int(obj.MaxQ, 0.0, 5e4, 0.0, 1.0),
                       unit=1)  # q upper bound

    '''result.upper_bound(v, obj.vmax, unit=obj.vmax)  # v upper bound
    result.upper_bound(chi, obj.chimax, unit=obj.chimax)  # chi upper bound
    result.upper_bound(gamma, obj.gammamax, unit=obj.gammamax)  # gamma upper bound
    result.upper_bound(teta, obj.tetamax, unit=obj.tetamax)  # teta upper bound
    result.upper_bound(lam, obj.lammax, unit=obj.lammax)  # lam upper bound
    result.upper_bound(h, obj.hmax, unit=obj.hmax)  # h upper bound
    result.upper_bound(m, obj.M0, unit=obj.M0)  # m upper bound
    result.upper_bound(alfa, obj.alfamax, unit=obj.alfamax)  # alfa upper bound
    result.upper_bound(delta, obj.deltamax, unit=obj.deltamax)  # delta upper bound
    result.upper_bound(az, obj.MaxAz, unit=obj.MaxAz)  # az upper bound
    result.upper_bound(ax, obj.MaxAx, unit=obj.MaxAx)  # ax upper bound
    result.upper_bound(q, obj.MaxQ, unit=obj.MaxQ)  # q upper bound'''

    return result()


def dynamicsVel(states, contr, obj):
    '''this functions receives the states and controls unscaled and calculates the dynamics'''
    v = states[0]
    chi = states[1]
    gamma = states[2]
    # teta = states[3]
    lam = states[4]
    h = states[5]
    m = states[6]
    alfa = contr[0]
    delta = contr[1]
    deltaf = 0.0 #contr[2]
    tau = 0.0 #contr[2]
    mu = 0.0 #contr[4]

    Press, rho, c = obj.isa(h, obj.psl, obj.g0, obj.Re, 1)

    M = v / c

    L, D, MomA = obj.aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                                obj.xcg0, obj.xcgf, obj.pref, 1)

    T, Deps, isp, MomT = obj.thrust(Press, m, presv, spimpv, delta, tau, 1, obj.psl, obj.M0, obj.m10,
                                    obj.lRef, obj.xcgf, obj.xcg0)

    eps = Deps + alfa
    g0 = obj.g0

    if h == 0:
        g = g0
    else:
        g = obj.g0 * (obj.Re / (obj.Re + h)) ** 2
    # g = np.asarray(g, dtype=np.float64)

    if h < obj.hvert:
        dx = np.array(((T*np.cos(eps)-D)/m-g, 0, 0, 0, 0, v, -T / (g0 * isp)))
    else:
        dx = np.array((((T * np.cos(eps) - D) / m) - g * np.sin(gamma) + (obj.omega ** 2) * (obj.Re + h) * np.cos(lam) * \
                    (np.cos(lam) * np.sin(gamma) - np.sin(lam) * np.cos(gamma) * np.sin(chi)),
                    ((T * np.sin(eps) + L) * np.sin(mu)) / (m * v * np.cos(gamma)) - np.cos(gamma) * np.cos(chi) *
                    np.tan(lam) * (v / (obj.Re + h)) + 2 * obj.omega * (
                               np.cos(lam) * np.tan(gamma) * np.sin(chi) - np.sin(lam)) \
                    - (obj.omega ** 2) * ((obj.Re + h) / (v * np.cos(gamma))) * np.cos(lam) * np.sin(lam) * np.cos(
                           chi),
                    ((T * np.sin(eps) + L) * np.cos(mu)) / (m * v) - (g / v - v / (obj.Re + h)) * np.cos(
                           gamma) + 2 * obj.omega \
                    * np.cos(lam) * np.cos(chi) + (obj.omega ** 2) * ((obj.Re + h) / v) * np.cos(lam) * \
                    (np.sin(lam) * np.sin(gamma) * np.sin(chi) + np.cos(lam) * np.cos(gamma)),
                    -np.cos(gamma) * np.cos(chi) * (v / ((obj.Re + h) * np.cos(lam))),
                    np.cos(gamma) * np.sin(chi) * (v / (obj.Re + h)),
                    v * np.sin(gamma),
                    -T / (g0 * isp)))
    return dx


if __name__ == '__main__':

    cl = np.array(fileReadOr("/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/coeff_files/clfile.txt"))
    cd = np.array(fileReadOr("/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/coeff_files/cdfile.txt"))
    cm = np.array(fileReadOr("/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/coeff_files/cmfile.txt"))
    # cl = np.load("/home/francesco/Desktop/PhD/FESTIP_Work/coeff_files/cl_smooth_few.npy")
    # cd = np.load("/home/francesco/Desktop/PhD/FESTIP_Work/coeff_files/cd_smooth_few.npy")
    # cm = np.load("/home/francesco/Desktop/PhD/FESTIP_Work/coeff_files/cm_smooth_few.npy")
    cl = np.reshape(cl, (13, 17, 6))
    # cd = np.asarray(cd)
    cd = np.reshape(cd, (13, 17, 6))
    # cm = np.asarray(cm)
    cm = np.reshape(cm, (13, 17, 6))


    with open("/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/coeff_files/impulse.dat") as f:
        impulse = []
        for line in f:
            line = line.split()
            if line:
                line = [float(i) for i in line]
                impulse.append(line)

    f.close()

    presv = []
    spimpv = []

    for i in range(len(impulse)):
        presv.append(impulse[i][0])
        spimpv.append(impulse[i][1])

    presv = np.asarray(presv)
    spimpv = np.asarray(spimpv)
    spimp_interp = splrep(presv, spimpv, s=2)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    flag_savefig = True
    source = 'matlab'
    if source == 'matlab':
        mat_contents = sio.loadmat( '/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/workspace_init_cond.mat')
        tfin = mat_contents['t'][0][-1]
    else:
        tfin = np.load('/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/Collocation_Algorithm/nice_initCond/Data_timeTot.npy')[-1]

    pool = Pool(processes=3)
    plt.ion()
    start = time.time()
    n = [100]
    time_init = [0.0, tfin]
    num_states = [7]
    num_controls = [2]
    max_iteration = 1
    Ncontrols = num_controls[0]
    Nstates = num_states[0]
    Npoints = n[0]
    varStates = Nstates * Npoints
    varTot = (Nstates + Ncontrols) * Npoints
    Nint = 1000
    maxiter = 10000
    ftol = 1e-8

    if flag_savefig:
        os.makedirs("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr))
        savefig_file = "/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/Res_".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr)
    # -------------
    # set OpenGoddard class for algorithm determination
    prob = Problem(time_init, n, num_states, num_controls, max_iteration)

    # -------------
    # create instance of operating object
    obj = Spaceplane()

    unit_v = obj.vmax
    unit_chi = obj.chimax
    unit_gamma = obj.gammamax
    unit_teta = obj.tetamax
    unit_lam = obj.lammax
    unit_h = obj.hmax
    unit_m = obj.M0
    unit_t = 700
    unit_alfa = np.deg2rad(40)
    unit_delta = 1
    #unit_deltaf = np.deg2rad(30)
    #unit_tau = 1
    #unit_mu = np.deg2rad(60)
    prob.set_unit_states_all_section(0, unit_v)
    prob.set_unit_states_all_section(1, unit_chi)
    prob.set_unit_states_all_section(2, unit_gamma)
    prob.set_unit_states_all_section(3, unit_teta)
    prob.set_unit_states_all_section(4, unit_lam)
    prob.set_unit_states_all_section(5, unit_h)
    prob.set_unit_states_all_section(6, unit_m)
    prob.set_unit_controls_all_section(0, unit_alfa)
    prob.set_unit_controls_all_section(1, unit_delta)
    #prob.set_unit_controls_all_section(2, unit_deltaf)
    #prob.set_unit_controls_all_section(2, unit_tau)
    #prob.set_unit_controls_all_section(4, unit_mu)
    prob.set_unit_time(unit_t)

    # =================
    # initial parameters guess
    # velocity

    states_init, controls_init = init_conds(prob.time_update(), source)

    v_init = states_init[0]
    chi_init = states_init[1]
    gamma_init = states_init[2]
    teta_init = states_init[3]
    lam_init = states_init[4]
    h_init = states_init[5]
    m_init = states_init[6]
    alfa_init = controls_init[0]
    delta_init = controls_init[1]
    '''v_init = Guess.linear(prob.time_all_section, 1, obj.Vtarget)
    chi_init = Guess.linear(prob.time_all_section, obj.chistart, obj.chi_fin)
    gamma_init = Guess.linear(prob.time_all_section, obj.gammastart, 0.0)
    teta_init = Guess.constant(prob.time_all_section, obj.longstart)
    lam_init = Guess.constant(prob.time_all_section, obj.latstart)
    h_init = Guess.linear(prob.time_all_section, 1, obj.Hini)
    m_init = Guess.linear(prob.time_all_section, obj.M0, obj.m10)

    #alfa_init = Guess.zeros(prob.time_all_section)
    part1 = np.repeat(1.0, obj.n32)
    part2 = Guess.linear(prob.time_all_section[obj.n32:], 1.0, 0.001)
    delta_init = np.hstack((part1, part2))'''
    #deltaf_init = Guess.zeros(prob.time_all_section)
    #tau_init = Guess.zeros(prob.time_all_section)
    #mu_init = Guess.zeros(prob.time_all_section)

    # ===========
    # Substitution initial value to parameter vector to be optimized
    # non dimensional values (Divide by scale factor)

    prob.set_states_all_section(0, v_init)
    prob.set_states_all_section(1, chi_init)
    prob.set_states_all_section(2, gamma_init)
    prob.set_states_all_section(3, teta_init)
    prob.set_states_all_section(4, lam_init)
    prob.set_states_all_section(5, h_init)
    prob.set_states_all_section(6, m_init)
    prob.set_controls_all_section(0, alfa_init)
    prob.set_controls_all_section(1, delta_init)
    #prob.set_controls_all_section(2, deltaf_init)
    #prob.set_controls_all_section(2, tau_init)
    #prob.set_controls_all_section(4, mu_init)

    # ========================
    # Main Process
    # Assign problem to SQP solver
    prob.dynamics = [dynamics]
    # prob.knot_states_smooth = []
    prob.cost = cost
    prob.equality = equality
    prob.inequality = inequality


    # prob.cost_derivative = cost_derivative
    # prob.ineq_derivative = ineq_derivative
    # prob.eq_derivative = eq_derivative

    def display_func():

        m = prob.states_all_section(6)
        h = prob.states_all_section(5)
        #time = prob.time_update()
        #part1 = np.repeat(1.0, int(len(m) * 0.4))
        #part2 = Guess.linear(time[int(len(m) * 0.4):], 1.0, 0.0001)
        #delta = np.hstack((part1, part2))
        delta = prob.controls_all_section(1)
        tau = np.zeros(len(m)) #prob.controls_all_section(2)

        tf = prob.time_final(-1)

        Press, rho, c = obj.isa(h, obj.psl, obj.g0, obj.Re, 0)

        T, Deps, isp, MomT = obj.thrust(Press, m, presv, spimpv, delta, tau, prob.nodes[0], obj.psl, obj.M0, obj.m10,
                                        obj.lRef, obj.xcgf, obj.xcg0)

        # Hohmann transfer mass calculation
        r1 = h[-1] + obj.Re
        Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
        Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
        mf = m[-1] / (np.exp(Dv1 / obj.gIsp) * np.exp(Dv2 / (obj.g0 * isp[-1])))

        print("m0          : {0:.5f}".format(m[0]))
        print("m before Ho : {0:.5f}".format(m[-1]))
        print("mf          : {0:.5f}".format(mf))
        print("altitude Hohmann starts: {0:.5f}".format(h[-1]))
        print("final time  : {0:.3f}".format(tf))


    prob.solve(obj, display_func, ftol=ftol, maxiter=maxiter)

    end = time.time()
    time_elapsed = end - start
    tformat = str(datetime.timedelta(seconds=int(time_elapsed)))
    print("Time elapsed:for total optimization ", tformat)


    def dynamicsInt(t, states, alfa_int, delta_int):#, deltaf_int, tau_int, mu_int):
        # this functions receives the states and controls unscaled and calculates the dynamics

        v = states[0]
        chi = states[1]
        gamma = states[2]
        # teta = states[3]
        lam = states[4]
        h = states[5]
        m = states[6]
        alfa = alfa_int(t)
        delta = delta_int(t)
        deltaf = 0.0 #deltaf_int(t)
        tau = 0.0 #tau_int(t)
        mu = 0.0 #mu_int(t)

        # if h<0:
        #   print("h: ", h, "gamma: ", gamma, "v: ", v)

        Press, rho, c = obj.isa(h, obj.psl, obj.g0, obj.Re, 1)

        M = v / c

        L, D, MomA = obj.aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                                    obj.xcg0, obj.xcgf, obj.pref, 1)

        T, Deps, isp, MomT = obj.thrust(Press, m, presv, spimpv, delta, tau, 1, obj.psl, obj.M0, obj.m10,
                                        obj.lRef, obj.xcgf, obj.xcg0)

        eps = Deps + alfa
        g0 = obj.g0

        if h == 0:
            g = g0
        else:
            g = obj.g0 * (obj.Re / (obj.Re + h)) ** 2

        if h < obj.hvert:
            dx = np.array(((T * np.cos(eps) - D) / m - g, 0, 0, 0, 0, v, -T / (g0 * isp)))
        else:
            dx = np.array(
                (((T * np.cos(eps) - D) / m) - g * np.sin(gamma) + (obj.omega ** 2) * (obj.Re + h) * np.cos(lam) * \
                 (np.cos(lam) * np.sin(gamma) - np.sin(lam) * np.cos(gamma) * np.sin(chi)),
                 ((T * np.sin(eps) + L) * np.sin(mu)) / (m * v * np.cos(gamma)) - np.cos(gamma) * np.cos(chi) *
                 np.tan(lam) * (v / (obj.Re + h)) + 2 * obj.omega * (
                         np.cos(lam) * np.tan(gamma) * np.sin(chi) - np.sin(lam)) \
                 - (obj.omega ** 2) * ((obj.Re + h) / (v * np.cos(gamma))) * np.cos(lam) * np.sin(lam) * np.cos(
                     chi),
                 ((T * np.sin(eps) + L) * np.cos(mu)) / (m * v) - (g / v - v / (obj.Re + h)) * np.cos(
                     gamma) + 2 * obj.omega \
                 * np.cos(lam) * np.cos(chi) + (obj.omega ** 2) * ((obj.Re + h) / v) * np.cos(lam) * \
                 (np.sin(lam) * np.sin(gamma) * np.sin(chi) + np.cos(lam) * np.cos(gamma)),
                 -np.cos(gamma) * np.cos(chi) * (v / ((obj.Re + h) * np.cos(lam))),
                 np.cos(gamma) * np.sin(chi) * (v / (obj.Re + h)),
                 v * np.sin(gamma),
                 -T / (g0 * isp)))
        dx = np.reshape(dx, (7,))
        return dx


    def SingleShooting(states, controls, dyn, time, Nint):

        x = np.zeros((Nint, Nstates))

        x[:Nstates] = states  # vector of intial states ready

        # now interpolation of controls

        alfa_Int = interpolate.PchipInterpolator(time, controls[0, :])
        delta_Int = interpolate.PchipInterpolator(time, controls[1, :])
        #deltaf_Int = interpolate.PchipInterpolator(time, controls[2, :])
        #tau_Int = interpolate.PchipInterpolator(time, controls[2, :])
        #mu_Int = interpolate.PchipInterpolator(time, controls[4, :])

        time_new = np.linspace(0, time[-1], Nint)

        dt = (time_new[1] - time_new[0])

        t = time_new

        # sol = solve_ivp(fun=lambda t, x: dyn(t, x, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int), t_span=[0, time[-1]], y0=x,
        # t_eval=time_new, method='RK45')

        for i in range(Nint - 1):
            # print(i, x[i,:])
            # print(u[i,:])
            k1 = dt * dyn(t[i], x[i, :], alfa_Int, delta_Int) #, deltaf_Int, tau_Int, mu_Int)
            # print("k1: ", k1)
            k2 = dt * dyn(t[i] + dt / 2, x[i, :] + k1 / 2, alfa_Int, delta_Int) #, deltaf_Int, tau_Int, mu_Int)
            # print("k2: ", k2)
            k3 = dt * dyn(t[i] + dt / 2, x[i, :] + k2 / 2, alfa_Int, delta_Int) #, deltaf_Int, tau_Int, mu_Int)
            # print("k3: ", k3)
            k4 = dt * dyn(t[i + 1], x[i, :] + k3, alfa_Int, delta_Int) #, deltaf_Int, tau_Int, mu_Int)
            # print("k4: ", k4)
            x[i + 1, :] = x[i, :] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # vres = sol.y[0, :]
        # chires = sol.y[1, :]
        # gammares = sol.y[2, :]
        # tetares = sol.y[3, :]
        # lamres = sol.y[4, :]
        # hres = sol.y[5, :]
        # mres = sol.y[6, :]
        vres = x[:, 0]
        chires = x[:, 1]
        gammares = x[:, 2]
        tetares = x[:, 3]
        lamres = x[:, 4]
        hres = x[:, 5]
        mres = x[:, 6]
        alfares = alfa_Int(time_new)
        deltares = delta_Int(time_new)
        deltafres = np.zeros(len(time_new)) #deltaf_Int(time_new)
        taures = np.zeros(len(time_new)) #tau_Int(time_new)
        mures = np.zeros(len(time_new)) #mu_Int(time_new)

        return vres, chires, gammares, tetares, lamres, hres, mres, time_new, alfares, deltares, deltafres, taures, mures


    # ======
    # Post Process
    # ------------------------
    # Convert parameter vector to variable

    # integration of collocation results
    # here angles are in radians and not scaled

    v = prob.states_all_section(0)
    chi = prob.states_all_section(1)
    gamma = prob.states_all_section(2)
    teta = prob.states_all_section(3)
    lam = prob.states_all_section(4)
    h = prob.states_all_section(5)
    m = prob.states_all_section(6)
    time = prob.time_update()
    alfa = prob.controls_all_section(0)
    #part1 = np.repeat(1.0, int(len(v) * 0.4))
    #part2 = Guess.linear(time[int(len(v) * 0.4):], 1.0, 0.0001)
    #delta = np.hstack((part1, part2))
    delta = prob.controls_all_section(1)
    deltaf = np.zeros(len(v)) #prob.controls_all_section(2)
    tau = np.zeros(len(v)) #prob.controls_all_section(2)
    mu = np.zeros(len(v)) #prob.controls_all_section(4)


    Uval = np.array((alfa, delta)) #np.vstack((delta))#, deltaf, tau, mu))

    Xinit = np.array((v[0], chi[0], gamma[0], teta[0], lam[0], h[0], m[0]))

    vres, chires, gammares, tetares, lamres, hres, mres, tres, alfares, deltares, deltafres, taures, mures = \
        SingleShooting(Xinit, Uval, dynamicsInt, time, Nint)

    Press, rho, c = obj.isa(h, obj.psl, obj.g0, obj.Re, 0)

    Pressres, rhores, cres = obj.isa(hres, obj.psl, obj.g0, obj.Re, 0)

    M = v / c

    Mres = vres / cres

    L, D, MomA = obj.aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10, obj.xcg0, obj.xcgf, obj.pref, n[0])

    Lres, Dres, MomAres = obj.aeroForces(Mres, alfares, deltafres, cd, cl, cm, vres, obj.wingSurf, rhores, obj.lRef, obj.M0, mres, obj.m10,
                                obj.xcg0, obj.xcgf, obj.pref, len(vres))

    T, Deps, isp, MomT = obj.thrust(Press, m, presv, spimpv, delta, tau, n[0], obj.psl, obj.M0, obj.m10, obj.lRef,
                                    obj.xcgf,
                                    obj.xcg0)

    Tres, Depsres, ispres, MomTres = obj.thrust(Pressres, mres, presv, spimpv, deltares, taures, len(vres), obj.psl, obj.M0, obj.m10, obj.lRef,
                                    obj.xcgf,obj.xcg0)

    MomTot = MomA + MomT
    MomTotres = MomAres + MomTres

    g0 = obj.g0
    eps = Deps + alfa

    # dynamic pressure

    q = 0.5 * rho * (v ** 2)
    qres = 0.5 * rhores * (vres ** 2)

    # accelerations

    ax = (T * np.cos(Deps) - D * np.cos(alfa) + L * np.sin(alfa)) / m
    az = (T * np.sin(Deps) + D * np.sin(alfa) + L * np.cos(alfa)) / m

    ax_res = (Tres * np.cos(Depsres) - Dres * np.cos(alfares) + Lres * np.sin(alfares)) / mres
    az_res = (Tres * np.sin(Depsres) + Dres * np.sin(alfares) + Lres * np.cos(alfares)) / mres

    r1 = h + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m / (np.exp(Dv1 / obj.gIsp) * np.exp(Dv2 / (obj.g0 * isp)))

    if flag_savefig:
        res = open("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/resColloc_{}_{}.txt".format(os.path.basename(__file__), n[0], maxiter, max_iteration, timestr, os.path.basename(__file__), timestr), "w")

        res.write("Max number Optimization iterations: " + str(max_iteration) + "\n" + "Number of NLP iterations: "
                  + str(maxiter) + "\n" + "v: " + str(v) + "\n" + "Chi: " + str(np.rad2deg(chi)) + "\n" + "Gamma: "
                  + str(np.rad2deg(gamma)) + "\n" + "Teta: " + str(np.rad2deg(teta)) + "\n" + "Lambda: "
                  + str(np.rad2deg(lam)) + "\n" + "Height: " + str(h) + "\n" + "Mass: " + str(m) + "\n" + "mf: "
                  + str(mf) + "\n" + "Objective Function: " + str(-mf / unit_m) + "\n" + "Alfa: " + str(np.rad2deg(alfa))
                  + "\n" + "Delta: " + str(delta) + "\n" + "Delta f: " + str(np.rad2deg(deltaf)) + "\n" + "Tau: "
                  + str(tau) + "\n" + "Mu: " + str(np.rad2deg(mu)) + "\n" + "Eps: " + str(np.rad2deg(eps)) + "\n" + "Lift: "
                  + str(L) + "\n" + "Drag: " + str(D) + "\n" + "Thrust: " + str(T) + "\n" + "Spimp: " + str(
            isp) + "\n" + "c: "
                  + str(c) + "\n" + "Mach: " + str(M) + "\n" + "Time vector: " + str(time) + "\n" + "Press: " + str(Press)
                  + "\n" + "Dens: " + str(rho) + "\n" + "Time elapsed for optimization: " + tformat + "\n")
        res.close()

        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/v".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), vres)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/chi".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), chires)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/gamma".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), gammares)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/teta".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), tetares)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/lambda".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), lamres)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/h".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), hres)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/m".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), mres)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/alfa".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), alfa)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/delta".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), delta)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/deltaf".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), deltaf)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/tau".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), tau)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/mu".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), mu)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/timeTot".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), tres)
        np.save("/home/francesco/Desktop/PhD/FESTIP_Work/Collocation_Algorithm/Results/Res{}_p{}_it{}x{}_{}/timeCol".format(
            os.path.basename(__file__), n[0], maxiter, max_iteration, timestr), time)

    # ------------------------
    # Visualization

    plt.figure(0)
    plt.title("Altitude profile")
    plt.plot(time, h / 1000, marker=".", label="Altitude")
    plt.plot(tres, hres / 1000, label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Altitude [km]")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "altitude" + ".png")

    plt.figure(1)
    plt.title("Velocity")
    plt.plot(time, v, marker=".", label="V")
    plt.plot(tres, vres, label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "velocity" + ".png")

    plt.figure(2)
    plt.title("Mass")
    plt.plot(time, m, marker=".", label="Mass")
    plt.plot(tres, mres, label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Mass [kg]")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "mass" + ".png")

    '''plt.figure(3)
    plt.title("Altitude profile")
    plt.plot(time, h/1000, marker=".", label="Altitude")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Altitude [km]")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "altitudeNONINT" + ".png")

    plt.figure(4)
    plt.title("Velocity")
    plt.plot(time, v, marker=".", label="V")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "velocityNONINT" + ".png")

    plt.figure(5)
    plt.title("Mass")
    plt.plot(time, m, marker=".", label="Mass")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Mass [kg]")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "massNONINT" + ".png")'''

    plt.figure(6)
    plt.title("Acceleration")
    plt.plot(time, ax, marker=".", label="Acc x")
    plt.plot(time, az, marker=".", label="Acc z")
    plt.plot(tres, ax_res, label="Acc x Integration")
    plt.plot(tres, az_res, label="Acc z Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Acceleration [m/s2]")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "acceleration" + ".png")

    plt.figure(7)
    plt.title("Throttle profile")
    plt.plot(time, delta, ".", label="Delta")
    plt.plot(tres, deltares, label="Interp")
    #plt.plot(time, tau, ".", label="Tau")
    #plt.plot(tres, taures, label="Interp")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" % ")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "Throttle" + ".png")

    plt.figure(8)
    plt.title("Angle of attack profile")
    plt.plot(time, np.rad2deg(alfa), ".", label="Alpha")
    plt.plot(tres, np.rad2deg(alfares), label="Interp")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "angAttack" + ".png")

    '''plt.figure(9)
    plt.title("Body Flap deflection profile")
    plt.plot(time, np.rad2deg(deltaf), ".", label="Delta f")
    plt.plot(tres, np.rad2deg(deltafres), label="Interp")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "bdyFlap" + ".png")

    plt.figure(10)
    plt.title("Bank angle profile")
    plt.plot(time, np.rad2deg(mu), ".", label="Mu")
    plt.plot(tres, np.rad2deg(mures), label="Interp")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "Roll" + ".png")'''

    '''
    plt.subplot(3,1,1)

    plt.step(time, delta, marker="o", label="Delta")
    plt.step(time, tau, marker="o", label="Tau")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" % ")
    plt.legend(loc="best")


    plt.subplot(3,1,2)
    plt.plot(time, np.rad2deg(alfa), marker="o", label="Alfa")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")


    plt.subplot(3,1,3)
    plt.plot(time, np.rad2deg(deltaf), marker="o", label="Deltaf")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")


    plt.subplot(2,2,4)
    plt.step(time, np.rad2deg(mu), where='post', marker=".", label="Mu")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")
    plt.suptitle("Controls profile")
    if flag_savefig:
        plt.savefig(savefig_file + "Commands" + ".png")

    '''

    plt.figure(11)
    plt.title("Trajectory angles")
    plt.plot(time, np.rad2deg(chi), marker=".", label="Chi")
    plt.plot(time, np.rad2deg(gamma), marker=".", label="Gamma")
    plt.plot(time, np.rad2deg(teta), marker=".", label="Theta")
    plt.plot(time, np.rad2deg(lam), marker=".", label="Lambda")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "Angles" + ".png")

    plt.figure(12)
    plt.title("Chi")
    plt.plot(time, np.rad2deg(chi), marker=".", label="Chi")
    plt.plot(tres, np.rad2deg(chires), label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "chi" + ".png")

    plt.figure(13)
    plt.title("Gamma")
    plt.plot(time, np.rad2deg(gamma), marker=".", label="Gamma")
    plt.plot(tres, np.rad2deg(gammares), label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "Gamma" + ".png")

    plt.figure(14)
    plt.title("Teta")
    plt.plot(time, np.rad2deg(teta), marker=".", label="Theta")
    plt.plot(tres, np.rad2deg(tetares), label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "teta" + ".png")

    plt.figure(15)
    plt.title("Lambda")
    plt.plot(time, np.rad2deg(lam), marker=".", label="Lambda")
    plt.plot(tres, np.rad2deg(lamres), label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "lam" + ".png")

    plt.figure(16)
    plt.title("Dynamic pressure profile")
    plt.plot(time, q / 1000, marker=".", label="Q")
    plt.plot(tres, qres / 1000, label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" kPa ")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "dynPress" + ".png")

    plt.figure(17)
    plt.title("Moment")
    plt.plot(time, MomTot / 1000, marker=".", label="Total Moment")
    plt.plot(tres, MomTotres / 1000, label="Integration")
    #plt.axhline(obj.k / 1000, 0, time[-1], color='r')
    #plt.axhline(-obj.k / 1000, 0, time[-1], color='r')
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" kNm ")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "MomTot" + ".png")

    plt.figure(19)
    plt.title("Mach profile")
    plt.plot(time, M, marker=".", label="Mach")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "mach" + ".png")

    plt.figure(20)
    plt.title("Lift, Drag and Thrust profile")
    plt.plot(time, L, marker=".", label="Lift")
    plt.plot(time, D, marker=".", label="Drag")
    plt.plot(time, T, marker=".", label="Thrust")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "LDT" + ".png")

    figs = plt.figure(21)
    ax = figs.add_subplot(111, projection='3d')
    ax.plot(np.rad2deg(teta), np.rad2deg(lam), h / 1e3, color='b', label="3d Trajectory")
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.set_zlabel('Altitude [km]')
    ax.legend()
    if flag_savefig:
        plt.savefig(savefig_file + "traj" + ".png")

    plt.show(block=True)
    pool.close()
    pool.join()
    '''plt.close(0)
    plt.close(1)
    plt.close(2)
    # plt.close(3)
    # plt.close(4)
    # plt.close(5)
    plt.close(6)
    plt.close(7)
    plt.close(8)
    plt.close(9)
    plt.close(10)
    plt.close(11)
    # plt.close(12)
    # plt.close(13)
    # plt.close(14)
    # plt.close(15)
    plt.close(16)
    plt.close(17)
    # plt.close(18)
    plt.close(19)
    plt.close(20)
    plt.close(21)'''