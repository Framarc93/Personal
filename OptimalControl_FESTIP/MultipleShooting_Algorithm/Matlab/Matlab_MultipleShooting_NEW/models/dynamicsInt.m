function dx = dynamicsInt(t, states, alfa_Int, delta_Int, obj, file)
%this functions receives the states and controls unscaled and calculates the dynamics%
global Pissue pen
v = states(1);
chi = states(2);
gamma = states(3);
teta = states(4);
lam = states(5);
h = states(6);
m = states(7);
alfa = ppval(alfa_Int,t);
delta = ppval(delta_Int,t);
deltaf = 0.0;% ppval(deltaf_Int, t);
tau = 0.0; %ppval(tau_Int, t);
mu = 0.0; %ppval(mu_Int, t);
if v > obj.vmax
    Pissue=1;
    pen = [pen; (v-obj.vmax)/obj.vmax];
elseif v < obj.vmin
    Pissue=1;    
    pen = [pen; (v-obj.vmin)/obj.vmax];
    %end
elseif isinf(v) || isnan(v)
    v = obj.vmax;
    Pissue=1;
    pen = [pen; 1];
    %end
end
if chi > obj.chimax
    Pissue=1;
    pen = [pen; (chi-obj.chimax)/obj.chimax];
elseif chi < obj.chimin 
    Pissue=1;    
    pen = [pen; (v-obj.chimin)/obj.chimax];
    %end
elseif isinf(chi) || isnan(chi)
    chi = obj.chimax;
    Pissue=1;
    pen = [pen; 1];
    %end
end
if gamma > obj.gammamax
    Pissue=1;
    pen = [pen; (gamma-obj.gammamax)/obj.gammamax];
elseif gamma < obj.gammamin 
    Pissue=1;    
    pen = [pen; (gamma-obj.gammamin)/obj.gammamax];
    %end
elseif isinf(gamma) || isnan(gamma)
    gamma = obj.gammamax;
    Pissue=1;
    pen = [pen; 1];
    %end
end
if h > obj.hmax
    Pissue=1;
    pen = [pen; (h-obj.hmax)/obj.hmax];
elseif h < obj.hmin 
    Pissue=1;    
    pen = [pen; (h-obj.hmin)/obj.hmax];
    %end
elseif isinf(h) || isnan(h)
    h = obj.hmax;
    Pissue=1;
    pen = [pen; 1];
    %end
end
% if isnan(sum(states))
%     Pissue = 1;
% if isnan(v)
% v=10;
% end
% if isnan(h)
% h=1;
% end
% if isnan(chi)
% chi=1;
% end
% if isnan(gamma)
% gamma=1;
% end
% if isnan(teta)
% teta=1;
% end
% if isnan(lam)
% lam=1;
% end
% if isnan(m)
% m=1;
% end
% end
% if gamma>deg2rad(89)
%     gamma=deg2rad(89);
%     Pissue = 1;
% end
% 
% if isinf(v)
%     v = 1e4;
%     Pissue = 1;
% end
% if isinf(h)
%     h = obj.hmax;
%     Pissue = 1;
% end
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

if t <= 2
    dx = [(T*cos(eps) - D)/m - g, 0, 0, 0, 0, v, -T/(g0*isp)];
else
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
end


