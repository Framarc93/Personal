function eq_cond = equality(var, conj, obj, prob, file)
global Controls

h = conj(end - 1);
lam = conj(end - 2);
stat = conj(end - prob.Nstates:end);
cont = Controls(end, :);
states_init = [0.000100000000000000,0.127777777777778,0.999444444444445,0.413611111111111,0.550387596899225,3.33333333333333e-06,1];
cont_init = [0.0476190476190476,1,0.400000000000000,0.500000000000000,0.500000000000000];

[vtAbs, chiass] = vass(stat, cont, obj.omega, obj, file);

vvv = sqrt(obj.GMe / (obj.Re + h));

if cos(obj.incl) / cos(lam) > 1
    chifin = pi;
else
    if cos(obj.incl) / cos(lam) < - 1
        chifin = 0.0;
    else
        chifin = 0.5 * pi + asin(cos(obj.incl) / cos(lam));
    end
end
[ir,ic]=size(conj(1:prob.Nstates * (prob.Nleg - 1)));
adV=max([abs(conj(1:prob.Nstates * (prob.Nleg - 1)));ones(ir,ic)]);
eq_cond = zeros(0);
eq_cond = [eq_cond, var(1:prob.Nstates) - states_init];
eq_cond = [eq_cond, (convStatestoOrig(var(prob.Nstates+1:prob.varStates), prob.Nstates, obj, prob.Nleg-1) - conj(1:prob.Nstates * (prob.Nleg - 1)))./adV];  % knotting conditions
eq_cond = [eq_cond, var(prob.varStates+1:prob.varStates + prob.Ncontrols) - cont_init];  % init cond on alpha
eq_cond = [eq_cond, vvv - vtAbs];
eq_cond = [eq_cond, chifin - chiass];
eq_cond = [eq_cond, conj(prob.varStates - 4)]; % final condition on gamma

end

