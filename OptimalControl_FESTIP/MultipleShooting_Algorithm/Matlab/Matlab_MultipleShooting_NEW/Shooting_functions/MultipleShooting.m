function [ineq_Cond, eq_Cond, objective] = MultipleShooting(var, prob, obj, file, states_init, cont_init)
%in this function the states and controls are scaled%
%this function takes the data from the optimization variable, so the angles enters in radians%
global varOld costOld eqOld ineqOld

ineq_Cond = [];
timestart = 0.0;

states_atNode = [];

i=0;
states_after = zeros(prob.Nint*prob.Nleg, prob.Nstates);
controls_after = zeros(prob.Nint*prob.Nleg, prob.Ncontrols);

varD = var.*(prob.UBV - prob.LBV) + prob.LBV;

while i <= prob.Nleg-1 
    controls = extract_controls(varD(prob.varStates+1+(prob.Ncontrols*prob.NContPoints)*i:prob.varStates+(i+1)*prob.Ncontrols*prob.NContPoints), prob.NContPoints, prob.Ncontrols);
    if i == 0
        states = states_init;
        states(2) = varD(1);
    else
        states = varD(1+i*prob.Nstates-(prob.Nstates-1):i*prob.Nstates+1);  
    end
    timeend = timestart + varD(prob.varTot + i+1);

    [vres, chires, gammares, tetares, lamres, hres, mres, t, alfares, deltares] = SingleShooting(states, controls, timestart, timeend, prob, obj, file);
    
    %vres(isnan(vres)) = 0;
    %chires(isnan(chires)) = 0;
    %gammares(isnan(gammares)) = 0;
    %tetares(isnan(tetares)) = 0;
    %lamres(isnan(lamres)) = 0;
    %hres(isnan(hres)) = 0;
    
    if i < prob.Nleg-1
        states_atNode = [states_atNode, [vres(end), chires(end), gammares(end), tetares(end), lamres(end), hres(end), mres(end)]]; 
    end
    t_ineqCond = linspace(0.0, timeend, prob.NineqCond);
    
    v_Int = pchip(t, vres);
    v_ineq = ppval(v_Int, t_ineqCond);
    chi_Int = pchip(t, chires);
    chi_ineq = ppval(chi_Int, t_ineqCond);
    gamma_Int = pchip(t, gammares);
    gamma_ineq = ppval(gamma_Int, t_ineqCond);
    teta_Int = pchip(t, tetares);
    teta_ineq = ppval(teta_Int, t_ineqCond);
    lam_Int = pchip(t, lamres);
    lam_ineq = ppval(lam_Int, t_ineqCond);
    h_Int = pchip(t, hres);
    h_ineq = ppval(h_Int, t_ineqCond);
    m_Int = pchip(t, mres);
    m_ineq = ppval(m_Int, t_ineqCond);
    alfa_Int_post = pchip(t, alfares);
    alfa_ineq = ppval(alfa_Int_post, t_ineqCond);
    delta_Int_post = pchip(t, deltares);
    delta_ineq = ppval(delta_Int_post, t_ineqCond);
   
    states_ineq = [v_ineq', chi_ineq', gamma_ineq', teta_ineq', lam_ineq', h_ineq', m_ineq'];
    states_after = [vres, chires, gammares, tetares, lamres, hres, mres];
    controls_after = [alfares, deltares];
    controls_ineq = [alfa_ineq', delta_ineq'];
    
    ineq_Cond = [ineq_Cond, inequalityAll(states_ineq, controls_ineq, prob.NineqCond, obj, file)];% evaluation of inequality constraints
    
    i = i + 1;
    timestart = timeend;
end


h = states_after(end, 6);
m = states_after(end, 7);
delta = controls_after(end, 2);
tau = 0.0; %controls_after(end, 4);

[Press, rho, c] = isa_FESTIP(h, obj);

[T, Deps, isp, MomT] = prop_FESTIP(Press, m, file.presv, file.spimpv, delta, tau, obj);

r1 = h + obj.Re;
Dv1 = sqrt(obj.GMe / r1) * (sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1);
Dv2 = sqrt(obj.GMe / obj.r2) * (1 - sqrt((2 * r1) / (r1 + obj.r2)));
mf = m / exp((Dv1 + Dv2) / (obj.g0 * isp));

ineq_Cond = [ineq_Cond, (obj.m10 - mf)/obj.M0, (6e4 - h)/obj.hmax];

eq_Cond = equality(varD, states_atNode, states_after, controls_after, obj, prob, file, states_init, cont_init);  % evaluation of equality constraints

objective = - mf / obj.M0;  % evaluation of objective function

costOld = objective;
ineqOld = ineq_Cond;
varOld = var;
eqOld = eq_Cond;
end





