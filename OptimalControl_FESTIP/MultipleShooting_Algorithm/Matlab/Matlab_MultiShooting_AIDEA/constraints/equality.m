function eq_cond = equality(var, conj, obj, prob, file)
global Controls
% v = states(1);
% chi = states(2);
% gamma = states(3);
% teta = states(4);
% lam = states(5);
% h = states(6);
% m = states(7);

hFinal = conj(end - 1);
lamFinal = conj(end - 2);
gammaFinal = conj(end - 4);
stat = conj(end - prob.Nstates+1:end);
cont = Controls(end, :);
states_init = [1,1.97222205475359,1.56905099754290,-0.921097512740007,0.0907571211037051,1,450400];
cont_init = [0.0,1.0,0.0,0.0,0.0];
for is=1:length(stat)
if isnan(stat(is))
stat(is)=1;
end
end
[vtAbs, chiass] = vass(stat, cont, obj.omega, obj, file);

vvv = sqrt(obj.GMe / (obj.Re + hFinal));

if cos(obj.incl) / cos(lamFinal) > 1
    chifin = pi;
else
    if cos(obj.incl) / cos(lamFinal) < - 1
        chifin = 0.0;
    else
        chifin = 0.5 * pi + asin(cos(obj.incl) / cos(lamFinal));
    end
end
[ir,ic]=size(conj(1:prob.Nstates * (prob.Nleg - 1)));
adV=repmat([100,1,1,1,1,100,1000],1,prob.Nleg - 1);
eq_cond = zeros(0);
eq_cond = [eq_cond, var(1:prob.Nstates) - states_init];
eq_cond = [eq_cond, (var(prob.Nstates+1:prob.varStates) - conj(1:prob.Nstates * (prob.Nleg - 1)))./adV];  % knotting conditions
%eq_cond = [eq_cond, var(prob.varStates+1:prob.varStates + prob.Ncontrols) - cont_init];  % init cond on alpha
eq_cond = [eq_cond, (vvv - vtAbs)/vvv];
eq_cond = [eq_cond, (chifin - chiass)/chifin];
eq_cond = [eq_cond, gammaFinal]; % final condition on gamma

end

