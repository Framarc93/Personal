clc
clearvars
close all
profile off
addpath(genpath('./'))

global dataS
dataS=[];
global bbb
bbb.f=1e16;
bbb.x=[];
bbb.xd=[];

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

v_init = interp1([tnew(1), tnew(end)],[1.0, obj.Vtarget], tnew);
chi_init = interp1([tnew(1), tnew(end)], [obj.chistart, obj.chi_fin], tnew);
gamma_init = interp1([tnew(1), tnew(end)], [obj.gammastart, 0.0], tnew);
teta_init = ones(1, prob.Nbar) * obj.longstart;
lam_init = ones(1, prob.Nbar) * obj.latstart;
h_init = interp1([tnew(1), tnew(end)], [1.0, obj.Hini], tnew);
m_init = interp1([tnew(1), tnew(end)], [obj.M0, obj.m10], tnew);
alfa_init = zeros(1, prob.varC);
delta_init = interp1([tcontr(1), tcontr(end)], [1.0, 0.001], tcontr);
deltaf_init = zeros(1, prob.varC);
tau_init = zeros(1, prob.varC);
mu_init = zeros(1, prob.varC);

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

X0d = [X, U, dt];  % vector of initial conditions here all the angles are in degrees!!!!!

LbS = [0.5,  deg2rad(110), deg2rad(88),    deg2rad(-53), deg2rad(4.8), 0.5,  obj.M0-10, ... 
        10, deg2rad(100), deg2rad(50),    deg2rad(-60), deg2rad(2.0), 2e4,  2.5e5, ...
       1000, deg2rad(100), deg2rad(0), deg2rad(-60), deg2rad(2.0),5e4,  1.5e5];

UbS = [1.5,  deg2rad(115), deg2rad(89.99), deg2rad(-51), deg2rad(5.8), 1.5, obj.M0+10, ... 
       2000, deg2rad(150), deg2rad(60), deg2rad(-45), deg2rad(8.0), 3e4,  3.5e5, ...
       4000,  deg2rad(150), deg2rad(20), deg2rad(-45), deg2rad(15.0),8e4, 2e5];

LbC = [deg2rad(-2.0), 0.8, deg2rad(-20.0), -0.1, deg2rad(-2), ...
        deg2rad(-2.0), 0.8, deg2rad(-20.0), -0.2, deg2rad(-30), ...
        deg2rad(-2.0), 0.7, deg2rad(-20.0), -0.2, deg2rad(-30), ...
        deg2rad(-2.0), 0.5, deg2rad(-10.0), -0.2, deg2rad(-30), ...
        deg2rad(-2.0), 0.1, deg2rad(-20.0), -0.2, deg2rad(-30), ...
        deg2rad(-2.0), 0.1, deg2rad(-20.0), -0.2, deg2rad(-30), ...
        deg2rad(-2.0), 0.1, deg2rad(-20.0), -0.2, deg2rad(-30), ...
        deg2rad(-2.0), 0.1, deg2rad(-20.0), -0.2, deg2rad(-30), ...
        deg2rad(-2.0), 0.1, deg2rad(-20.0), -0.2, deg2rad(-30), ...
        deg2rad(-2.0), 0.1, deg2rad(-20.0), -0.1, deg2rad(-30), ...
        deg2rad(-2.0), 0.001, deg2rad(-20.0), -0.1, deg2rad(-30), ...
        deg2rad(-2.0), 0.001, deg2rad(-20.0), -0.1, deg2rad(-30), ...
        deg2rad(-2.0), 0.001, deg2rad(-20.0), -0.1, deg2rad(-30.0), ...
        deg2rad(-2.0), 0.001, deg2rad(-20.0), -0.1, deg2rad(-30), ...
        deg2rad(-2.0), 0.001, deg2rad(-20.0), -0.1, deg2rad(-30)];

UbC = [ deg2rad(2.0),  1.0, deg2rad(30.), 0.1, deg2rad(2), ...
        deg2rad(2.0),  1.0, deg2rad(30.0), 0.2, deg2rad(30), ...
        deg2rad(10.0), 1.0, deg2rad(30.0), 0.2, deg2rad(30), ...
        deg2rad(10.0), 1.0, deg2rad(10.0), 0.2, deg2rad(30), ...
        deg2rad(10.0), 1.0, deg2rad(30.0), 0.2, deg2rad(30), ...
        deg2rad(10.0), 1.0, deg2rad(30.0), 0.2, deg2rad(30), ...
        deg2rad(10.0), 1.0, deg2rad(30.0), 0.2, deg2rad(30), ...
        deg2rad(10.0), 1.0, deg2rad(30.0), 0.1, deg2rad(30), ...
        deg2rad(10.0), 1.0, deg2rad(30.0), 0.1, deg2rad(30), ...
        deg2rad(10.0), 1.0, deg2rad(30.0), 0.1, deg2rad(30), ...
        deg2rad(10.0), 1.0, deg2rad(30.0), 0.1, deg2rad(30.0), ...
        deg2rad(10.0), 1.0, deg2rad(30.0), 0.1, deg2rad(30.0), ...
        deg2rad(10.0), 1.0, deg2rad(30.0), 0.1, deg2rad(30.0), ...
        deg2rad(10.0), 1.0, deg2rad(30.0), 0.1, deg2rad(30.0), ...
        deg2rad(5.0), 1.0, deg2rad(30.0), 0.1, deg2rad(30)];

Tlb = [60]; % time lower bounds
Tub = [300]; % time upper bounds
prob.LBV = [LbS, LbC, repmat(Tlb, 1, prob.Nleg)];
prob.UBV = [UbS, UbC, repmat(Tub, 1, prob.Nleg)];
X0a = (X0d - prob.LBV)./(prob.UBV-prob.LBV);
TlB = [0.0]; % time lower bounds
TuB = [1.0]; % time upper bounds

lb = [0.0];
ub = [1.0];

LB = [repmat(lb, 1, prob.Nleg*prob.Nstates), repmat(lb, 1, prob.Nleg * prob.NContPoints*prob.Ncontrols), repmat(TlB, 1, prob.Nleg)];
UB = [repmat(ub, 1, prob.Nleg*prob.Nstates), repmat(ub, 1, prob.Nleg * prob.NContPoints*prob.Ncontrols), repmat(TuB, 1, prob.Nleg)];
global varOld costOld eqOld ineqOld States Controls globTime
varOld = zeros(size(X0a));
costOld = 0;
eqOld = zeros(0);
ineqOld = zeros(0);
States = zeros(0);
Controls = zeros(0);
globTime = zeros(0);
% load('var_1.mat');
% for i=1:size(var_1,2)
%     y0(1,i)=0.001*(rand(1,1)*2.-1.)+var_1(1,i);
% end
% y0=max([y0;zeros(size(y0))]);
% y0=min([y0;ones(size(y0))]);
% X0=y0;
addpath(genpath('/home/kqb12101/WORK/TANDEM/AIDEA_orig'))

% ll=[5475,2.5,0,0,20,20,20,20,        0.0100000000000000,0.0100000000000000,0.0100000000000000,0.0100000000000000,1.05000000000000,1.05000000000000,1.05000000000000,-pi,-pi,-pi
%     9132,4.9,1,1,2500,2500,2500,2500,0.990000000000000,0.990000000000000,0.990000000000000,0.990000000000000,10,10,10,pi,pi,pi];
% ll=[6000,2.5,0,0,20,20,20,20,
% 0.0100000000000000,0.0100000000000000,0.0100000000000000,0.0100000000000000,1.05000000000000,1.05000000000000,1.05000000000000,-pi,-pi,-pi;
% 6500,3  ,1,1,2500,2500,2500,2500,0.990000000000000,0.990000000000000,0.990000000000000,0.990000000000000,10,10,10,pi,pi,pi];


global dataBest
dataBest=[];


Fobj       = @(x)mask_MS(x, obj, prob, file);

Options.P6=0.3;
Options.test_case=Fobj;
Options.NP0=40;
Options.itermax=1e5;
Options.F=.75;
Options.CR=.75;
Options.strategy0=6;
Options.percconv=0.2;
Options.nrloc=10;
Options.expsteploc1=0.01;
Options.expstepglo=0.1;
Options.expsteplocglo=4;
Options.vsave=[3e8 5e8 1e9 1.5e9];
Options.pdim=99;
Options.VTR=-1e10;
Options.saveFile='pippo.txt';
Options.ll=[LB; UB];
%        run_IDEA_newP5Rdcrf_2S(test_case,nTR_min,nTR_max,NP0,itermax,F,CR,strategy0,percconv,nrloc,expsteploc1,expstepglo,expsteplocglo,vsave,pdim)
%bestsol= run_IDEA_newP5Rdcrf_2S(Fobj,1,1,Npop,1e5,.65,.65,6,0.2,Nresloc,deltaloc,.1,4,[3e8 5e8 1e9 1.5e9],pdim);
load('bSol.mat', 'bSol')
X0a = (bSol.xd - prob.LBV)./(prob.UBV-prob.LBV);
save X0a X0a
bestsol= run_IDEA_newP5Rdcrf_2S_O(1,1,Options);










Plot_MS(opt, prob, obj, file);
save sol_N100_2