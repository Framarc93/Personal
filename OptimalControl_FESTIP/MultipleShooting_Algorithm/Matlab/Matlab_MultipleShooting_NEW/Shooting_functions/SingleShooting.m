function [vres, chires, gammares, tetares, lamres, hres, mres, t, alfares, deltares] = SingleShooting(states, controls, timestart, timeend, prob, obj, file)

%this function integrates the dynamics equation over time.
%INPUT: states: states vector
%controls: controls matrix
%dyn: dynamic equations
%tstart: initial time
%tfin: final time
%Nint: unmber of integration steps%
%states and controls must be given with real values!!! not scaled!!!%
%needed a fixed step integrator%
%tstart and tfin are the initial and final time of the considered leg%

timeCont = linspace(timestart, timeend, prob.NContPoints);
% now interpolation of controls

alfa_Int = pchip(timeCont, controls(1, :));
delta_Int = pchip(timeCont, controls(2, :));
%deltaf_Int = pchip(timeCont, controls(3, :));
%tau_Int = pchip(timeCont, controls(4, :));
%mu_Int = pchip(timeCont, controls(5, :));

t = linspace(timestart, timeend, prob.Nint);

dt = t(2) - t(1);
x = zeros(prob.Nint, 7);
x(1,:) = states;
for i=1:prob.Nint-1
    k1 = dt*dynamicsInt(t(i), x(i, :), alfa_Int, delta_Int, obj, file);
    k2 = dt*dynamicsInt(t(i) + dt / 2, x(i, :) + k1 / 2, alfa_Int, delta_Int, obj, file);
    k3 = dt*dynamicsInt(t(i) + dt / 2, x(i, :) + k2 / 2, alfa_Int, delta_Int, obj, file);
    k4 = dt*dynamicsInt(t(i + 1), x(i, :) + k3, alfa_Int, delta_Int, obj, file);
    x(i + 1, :) = x(i, :) + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
end

vres = x(:, 1); % orig interavals
% if sum(isinf(vres))>0
% Pissue=1;
% iv=find(isinf(vres));
% vres(iv)=1e4;
% end
chires = x(:, 2);
% if sum(isnan(chires))>0
% Pissue=1;
% iv=find(isnan(chires));
% chires(iv)=120*pi/180;
% end
gammares = x(:, 3);
% if sum(isnan(gammares))>0
% Pissue=1;
% iv=find(isnan(gammares));
% gammares(iv)=10*pi/180;
% end
% if sum(isinf(gammares))>0
% Pissue=1;
% iv=find(isinf(gammares));
% gammares(iv)=89*pi/180;
% end
tetares = x(:, 4);
lamres = x(:, 5);
hres = x(:, 6);
% if sum(isnan(hres))>0
% Pissue=1;
% iv=find(isnan(hres));
% hres(iv)=100;
% end
% if sum(isinf(hres))>0
% Pissue=1;
% iv=find(isinf(hres));
% hres(iv)=obj.hmax;
% end
mres = x(:, 7);
alfares = ppval(alfa_Int, t)';
deltares = ppval(delta_Int, t)';
%deltafres = zeros(1, length(vres))'; %ppval(deltaf_Int, t);
%taures = zeros(1, length(vres))'; %ppval(tau_Int, t);
%mures = zeros(1, length(vres))'; %ppval(mu_Int, t);

end


