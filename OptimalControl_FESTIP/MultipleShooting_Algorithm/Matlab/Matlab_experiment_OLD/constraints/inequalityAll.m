function iC = inequalityAll(states, controls, varnum, obj, file)
%this function takes states and controls unscaled%
v = states(:, 1);
chi = states(:, 2);
gamma = states(:, 3);
h = states(:, 6);
m = states(:, 7);
alfa = controls(:, 1);
delta = controls(:, 2);
deltaf = controls(:, 3);
tau = controls(:, 4);  % tau back to [-1, 1] interval

if isnan(v)
    v(isnan(v)) = 100;
end
if isnan(h)
    h(isnan(h)) = 1000;
end

[Press, rho, c] = isaMulti(h, obj);

M = v' ./ c;

[L, D, MomA] = aeroForcesMulti(M, alfa, deltaf, file.cd, file.cl, file.cm, v, rho, m, obj, varnum);

[T, Deps, isp, MomT] = thrustMulti(Press, m, file.presv, file.spimpv, delta, tau, obj, varnum);

MomTot = MomA + MomT;

% dynamic pressure

q = 0.5 * rho .* (v' .^ 2);

% accelerations

ax = (T .* cos(Deps) - D .* cos(alfa') + L .* sin(alfa')) ./ m';
az = (T .* sin(Deps) + D .* sin(alfa') + L .* cos(alfa')) ./ m';

r1 = h(end) + obj.Re;
Dv1 = sqrt(obj.GMe / r1) * (sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1);
Dv2 = sqrt(obj.GMe / obj.r2) * (1 - sqrt((2 * r1) / (r1 + obj.r2)));
mf = m(end) / exp((Dv1 + Dv2) / (obj.g0*isp(end)));


iC = -[(obj.MaxAx - ax)/obj.MaxAx, (obj.MaxAz - az)/obj.MaxAz, (obj.MaxQ - q)/obj.MaxQ, (obj.k - MomTot)/obj.k, (MomTot + obj.k)/obj.k, (mf - obj.m10)/obj.M0];
%iC = [iC, (-deg2rad(89.99) - gamma')./deg2rad(89.99), (gamma' - deg2rad(89.99))./deg2rad(89.99)];
%iC = [iC, (deg2rad(90) - chi')./deg2rad(270), (chi' - deg2rad(270))./deg2rad(270)];
iC = [iC, -h'./150000];
end


    