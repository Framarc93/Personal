function [vres, chires, gammares, tetares, lamres, hres, mres, t, alfares, deltares, deltafres, taures, mures, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int] = SingleShooting(states, controls, tstart, tfin, prob, obj, file)
%this function integrates the dynamics equation over time.%
%INPUT: states: states vector
%controls: controls matrix
%dyn: dynamic equations
%tstart: initial time
%tfin: final time
%Nint: unmber of integration steps%
%states and controls must be given with real values!!! not scaled!!!%
%needed a fixed step integrator%
%tstart and tfin are the initial and final time of the considered leg%

timeCont = linspace(tstart, tfin, prob.NContPoints);
x = zeros(prob.Nint, prob.Nstates);
x(1,:) = states;  % vector of intial states ready


% now interpolation of controls

alfa_Int = pchip(timeCont, controls(1, :));
delta_Int = pchip(timeCont, controls(2, :));
deltaf_Int = pchip(timeCont, controls(3, :));
tau_Int = pchip(timeCont, controls(4, :));
mu_Int = pchip(timeCont, controls(5, :));

t = linspace(tstart, tfin, prob.Nint);

dt = t(2) - t(1);


for i=1:prob.Nint-1
    
    k1 = dt*dynamicsInt(t(i), x(i, :), alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int, obj, file);
    k2 = dt*dynamicsInt(t(i) + dt / 2, x(i, :) + k1 / 2, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int, obj, file);
    k3 = dt*dynamicsInt(t(i) + dt / 2, x(i, :) + k2 / 2, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int, obj, file);
    k4 = dt*dynamicsInt(t(i + 1), x(i, :) + k3, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int, obj, file);
    x(i + 1, :) = x(i, :) + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
end

vres = x(:, 1); % orig interavals
chires = x(:, 2);
gammares = x(:, 3);
tetares = x(:, 4);
lamres = x(:, 5);
hres = x(:, 6);
mres = x(:, 7);
alfares = ppval(alfa_Int, t);
deltares = ppval(delta_Int, t);
deltafres = ppval(deltaf_Int, t);
taures = ppval(tau_Int, t);
mures = ppval(mu_Int, t);

end


