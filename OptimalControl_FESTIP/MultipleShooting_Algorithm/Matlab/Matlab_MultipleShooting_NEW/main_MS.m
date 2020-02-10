clc
clear all
close all

obj = Spaceplane;
prob = Problem;
file = Files;

ff = load('/home/francesco/Desktop/PhD/Git_workspace/Personal/OptimalControl_FESTIP/workspace_init_cond.mat');
time_tot = ff.t(end); % initial time
tstat = linspace(0, time_tot, prob.Nbar);
t_cont_vects = [];
for i=1:prob.Nleg
    t_cont_vects = [t_cont_vects; linspace(tstat(i), tstat(i + 1), prob.NContPoints)];  % time vector used for interpolation of controls intial guess
end
    
tnew = linspace(0, time_tot, prob.Nbar);
% Load initial conditions %
[states_init, controls_init] = intial_conds(tstat, t_cont_vects);
v_init = states_init(1, :);
chi_init = states_init(2, :);
gamma_init = states_init(3, :);
teta_init = states_init(4, :);
lam_init = states_init(5, :);
h_init = states_init(6, :);
m_init = states_init(7, :);
alfa_init = controls_init(1, :);
delta_init = controls_init(2, :);

% set vector of initial conditions of states and controls
X = zeros(1,prob.varStates);
U = zeros(1,prob.varControls);

states_init = [v_init(1), chi_init(1), gamma_init(1), teta_init(1), lam_init(1), h_init(1), m_init(1)];
cont_init = [alfa_init(1), delta_init(1)];
X(1) = chi_init(1);
XGuess = [v_init(2:end); chi_init(2:end); gamma_init(2:end); teta_init(2:end); lam_init(2:end); h_init(2:end); m_init(2:end)];  % states initial guesses

UGuess = [alfa_init; delta_init];  % states initial guesses

k = 2;
for i = 1:prob.Nleg-1
    %creation of vector of states initial guesses%
    for j = 1:prob.Nstates
        X(k) = XGuess(j, i);
        k = k + 1;
    end
    
end

k = 1;
for i = 1:prob.varC
    %creation of vector of controls initial guesses%
    for j = 1:prob.Ncontrols
        U(k) = UGuess(j, i);
        k = k + 1;
    end
end         

dt = diff(tnew);

X0d = [X, U, dt];  % vector of initial conditions here all the angles are in degrees!!!!!

Tlb = 20; % time lower bounds
Tub = 300; % time upper bounds
UbS = [obj.chimax, repmat([obj.vmax, obj.chimax, obj.gammamax, obj.tetamax, obj.lammax, obj.hmax, obj.M0], 1, prob.Nleg-1)];
LbS = [obj.chimin, repmat([obj.vmin, obj.chimin, obj.gammamin, obj.tetamin, obj.lammin, obj.hmin, obj.m10], 1, prob.Nleg-1)];
UbC = repmat([obj.alfamax, obj.deltamax], 1, prob.Nleg * prob.NContPoints);
LbC = repmat([obj.alfamin, obj.deltamin], 1, prob.Nleg * prob.NContPoints);
prob.LBV = [LbS, LbC, repmat(Tlb, 1, prob.Nleg)];
prob.UBV = [UbS, UbC, repmat(Tub, 1, prob.Nleg)];

X0a = (X0d - prob.LBV)./(prob.UBV-prob.LBV);

lb = 0.0;
ub = 1.0;

LB = zeros(1, (prob.Nleg-1)*prob.Nstates + 1 + prob.Nleg * prob.NContPoints  * prob.Ncontrols + prob.Nleg);
UB = ones(1, (prob.Nleg-1)*prob.Nstates + 1 + prob.Nleg * prob.NContPoints  * prob.Ncontrols + prob.Nleg);
global varOld costOld eqOld ineqOld States Controls globTime
varOld = zeros(size(X0a));
costOld = 0;
eqOld = zeros(0);
ineqOld = zeros(0);
States = zeros(0);
Controls = zeros(0);
globTime = zeros(0);
[opt, fval] = opti(obj, prob, file, X0a, LB, UB, states_init, cont_init);
Plot_MS(opt, prob, obj, file, states_init);
