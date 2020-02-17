function dx = dynamicsVel(states, controls, obj, file)
%this functions receives the states and controls unscaled and calculates the dynamics%
v = states(1);
chi = states(2);
gamma = states(3);
teta = states(4);
lam = states(5);
h = states(6);
m = states(7);
alfa = controls(1);
delta = controls(2);
deltaf = controls(3);
tau = controls(4);
mu = 0.0 ;%controls(5);


if isnan(v)
    v = 0.0;
elseif isinf(v) || v>1e6
    v = 1e6;
end
if isnan(chi)
    chi = 0.0;
elseif isinf(chi) || chi>obj.chimax
    chi = obj.chimax;
end
if isnan(gamma)
    gamma = 0.0;
elseif isinf(gamma)
    gamma = 1e10;
end
if isnan(lam)
    lam = 0.0;
elseif isinf(lam)
    lam = 1e10;
end
if isnan(h)
    h = 0.0;
elseif isinf(h) || h>1e7
    h = 1e7;
end
if isnan(m) || m <obj.m10
    m=obj.m10;
elseif isinf(m) || m> obj.M0
    m=obj.M0;
end

[Press, rho, c] = isa_FESTIP(h, obj);
M = v / c;
[L, D, MomA] = aero_FESTIP(M, alfa, deltaf, file.cd, file.cl, file.cm, v, rho, m, obj);
[T, Deps, isp, MomT] = prop_FESTIP(Press, m, file.presv, file.spimpv, delta, tau, obj);

eps = Deps + alfa;
g0 = obj.g0;
if h == 0
    g = g0;
else
    g = g0 * (obj.Re / (obj.Re + h)) ^ 2;
end

dx = [((T * cos(eps) - D) / m) - g * sin(gamma) + (obj.omega ^ 2) * ...
    (obj.Re + h) * cos(lam) * (cos(lam) * sin(gamma) - sin(lam) * cos(gamma) * sin(chi)), ...
    ((T * sin(eps) + L) * sin(mu)) / (m * v * cos(gamma)) - cos(gamma) * cos(chi) * ...
    tan(lam) * (v / (obj.Re + h)) + 2 * obj.omega * (cos(lam) * tan(gamma) * sin(chi) - sin(lam)) ...
    - (obj.omega ^ 2) * ((obj.Re + h) / (v * cos(gamma))) * cos(lam) * sin(gamma) * cos(chi), ...
    ((T * sin(eps) + L) * cos(mu)) / (m * v) - (g / v - v / (obj.Re + h)) * cos(gamma) + 2 * obj.omega ...
    * cos(lam) * cos(chi) + (obj.omega ^ 2) * ((obj.Re + h) / v) * cos(lam) * ...
    (sin(lam) * sin(gamma) * sin(chi) + cos(lam) * cos(gamma)), ...
    -cos(gamma) * cos(chi) * (v / ((obj.Re + h) * cos(lam))), ...
    cos(gamma) * sin(chi) * (v / (obj.Re + h)), ...
    v * sin(gamma), ...
    -T / (g0 * isp)];
end


