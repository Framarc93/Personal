function iC = inequalityAll(states, controls, varnum, obj, file)
%this function takes states and controls unscaled%

v = states(:, 1);
%chi = states(:, 2);
%gamma = states(:, 3);
h = states(:, 6);
m = states(:, 7);

alfa = controls(:, 1);
delta = controls(:, 2);
deltaf = zeros(1,length(alfa)); %controls(:, 3);
tau = zeros(1,length(alfa)); %controls(:, 4);  % tau back to [-1, 1] interval

[Press, rho, c] = isaMulti(h, obj);

M = v' ./ c;

[L, D, MomA] = aeroForcesMulti(M, alfa, deltaf, file.cd, file.cl, file.cm, v, rho, m, obj, varnum);

[T, Deps, isp, MomT] = thrustMulti(Press, m, file.presv, file.spimpv, delta, tau, obj, varnum);

%MomTot = MomA + MomT;
%MomTotA = abs(MomTot);

% dynamic pressure

q = 0.5 * rho .* (v' .^ 2);

% accelerations

ax = (T .* cos(Deps) - D .* cos(alfa') + L .* sin(alfa')) ./ m';
az = (T .* sin(Deps) + D .* sin(alfa') + L .* cos(alfa')) ./ m';

iC = -[(obj.MaxAx - ax)./obj.MaxAx, (obj.MaxAz - az)./obj.MaxAz, (obj.MaxAz + az)./obj.MaxAz, (obj.MaxQ - q)./obj.MaxQ];
end


    