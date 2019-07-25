function iC = inequalityAll(states, controls, varnum, obj, file)
%this function takes states and controls unscaled%
global Pissue

v = states(:, 1);
if sum(isnan(v))>0
Pissue=1;
iv=find(isnan(v));
v(iv)=100;
end

chi = states(:, 2);
if sum(isnan(chi))>0
Pissue=1;
iv=find(isnan(chi));
chi(iv)=120*pi/180;
end

gamma = states(:, 3);
if sum(isnan(gamma))>0
Pissue=1;
iv=find(isnan(gamma));
gamma(iv)=10*pi/180;
end

h = states(:, 6);
if sum(isnan(h))>0
Pissue=1;
iv=find(isnan(h));
h(iv)=100;
end
m = states(:, 7);
if sum(isnan(m))>0
Pissue=1;
iv=find(isnan(m));
m(iv)=100;
end

alfa = controls(:, 1);
delta = controls(:, 2);
deltaf = controls(:, 3);
tau = controls(:, 4);  % tau back to [-1, 1] interval

[Press, rho, c] = isaMulti(h, obj);

M = v' ./ c;

[L, D, MomA] = aeroForcesMulti(M, alfa, deltaf, file.cd, file.cl, file.cm, v, rho, m, obj, varnum);

[T, Deps, isp, MomT] = thrustMulti(Press, m, file.presv, file.spimpv, delta, tau, obj, varnum);

MomTot = MomA + MomT;
MomTotA = abs(MomTot);

% dynamic pressure

q = 0.5 * rho .* (v' .^ 2);

% accelerations

ax = (T .* cos(Deps) - D .* cos(alfa') + L .* sin(alfa')) ./ m';
az = (T .* sin(Deps) + D .* sin(alfa') + L .* cos(alfa')) ./ m';

r1 = h(end) + obj.Re;
Dv1 = sqrt(obj.GMe / r1) * (sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1);
Dv2 = sqrt(obj.GMe / obj.r2) * (1 - sqrt((2 * r1) / (r1 + obj.r2)));
mf = m(end) / exp((Dv1 + Dv2) / (obj.g0*isp(end)));


iC = -[(obj.MaxAx - ax)/obj.MaxAx, (obj.MaxAz - az)/obj.MaxAz, (obj.MaxQ - q)/obj.MaxQ, (obj.k - MomTot)/obj.k, (MomTot+obj.k)/obj.k, (mf - obj.m10)/obj.m10]';
%iC = [iC; (chi-270*pi/180); (90*pi/180-chi); 10*(gamma-90*pi/180); (-10*pi/180-gamma)];
end


    