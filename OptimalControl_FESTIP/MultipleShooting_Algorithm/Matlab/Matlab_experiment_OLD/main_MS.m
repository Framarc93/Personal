clc
clearvars
close all

obj = Spaceplane;
prob = Problem;
file = Files;

time_tot = 350; % initial time
tnew = linspace(0, time_tot, prob.Nbar);  % time vector used for interpolation of states initial guess
tcontr = linspace(0, time_tot, prob.varC);  % time vector used for interpolation of controls intial guess

%definiton of initial conditions%

% set vector of initial conditions of states and controls
X = zeros(1,prob.varStates);
U = zeros(1,prob.varControls);

%v_init = linear_fun(tnew, v_toNew(1.0), v_toNew(obj.Vtarget));

v_init = interp1([tnew(1), tnew(end)],[v_toNew(1.0), v_toNew(obj.Vtarget)], tnew);
chi_init = interp1([tnew(1), tnew(end)], [chi_toNew(obj.chistart), chi_toNew(obj.chi_fin)], tnew);
gamma_init = interp1([tnew(1), tnew(end)], [gamma_toNew(obj.gammastart), gamma_toNew(0.0)], tnew);
teta_init = ones(1, prob.Nbar) * teta_toNew(obj.longstart);
lam_init = ones(1, prob.Nbar) * lam_toNew(obj.latstart, obj);
h_init = interp1([tnew(1), tnew(end)], [h_toNew(1.0), h_toNew(obj.Hini)], tnew);
m_init = interp1([tnew(1), tnew(end)], [m_toNew(obj.M0, obj), m_toNew(obj.m10, obj)], tnew);
alfa_init = ones(1, prob.varC) * alfa_toNew(0.0);
delta_init = interp1([tcontr(1), tcontr(end)], [1.0, 0.001], tcontr);
deltaf_init = ones(1, prob.varC) * deltaf_toNew(0.0);
tau_init = ones(1, prob.varC) * tau_toNew(0.0);
mu_init = ones(1, prob.varC) * mu_toNew(0.0);

states_init = [v_init(1), chi_init(1), gamma_init(1), teta_init(1), lam_init(1), h_init(1), m_init(1)];
cont_init = [alfa_init(1), delta_init(1), deltaf_init(1), tau_init(1), mu_init(1)];

XGuess = [v_init; chi_init; gamma_init; teta_init; lam_init; h_init; m_init];  % states initial guesses

UGuess = [alfa_init; delta_init; deltaf_init; tau_init; mu_init];  % states initial guesses

k = 1;
for i = 1:prob.Nleg
  
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

dt = zeros(1,prob.Nleg);

for i = 1:prob.Nleg
    %creation of vector of time intervals%
    dt(i) = tnew(i + 1) - tnew(i);
end

X0 = [X, U, dt / prob.unit_t];  % vector of initial conditions here all the angles are in degrees!!!!!


% X0 has first all X guesses and then all U guesses
% at this point the vectors of initial guesses for states and controls for every time interval are defined

Xlb = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; % states lower bounds
Xub = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; % states upper bounds

Ulb = [0.0, 0.0001, 0.0, 0.0, 0.0]; % controls lower bounds
Uub = [1.0, 1.0, 1.0, 1.0, 1.0]; % controls upper bounds

Tlb = [1/prob.unit_t]; % time lower bounds
Tub = [1.0]; % time upper bounds


LB = [repmat(Xlb, 1, prob.Nleg), repmat(Ulb, 1, prob.Nleg * prob.NContPoints), repmat(Tlb, 1, prob.Nleg)];
UB = [repmat(Xub, 1, prob.Nleg), repmat(Uub, 1, prob.Nleg * prob.NContPoints), repmat(Tub, 1, prob.Nleg)];


global varOld costOld eqOld ineqOld States Controls globTime
varOld = zeros(size(X0));
costOld = 0;
eqOld = zeros(0);
ineqOld = zeros(0);
States = zeros(0);
Controls = zeros(0);
globTime = zeros(0);

[opt, fval] = opti(obj, prob, file, X0, LB, UB);
Plot(opt, prob, obj, file);
