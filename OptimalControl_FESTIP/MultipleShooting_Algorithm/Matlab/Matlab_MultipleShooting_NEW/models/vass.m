function [vela, chiass] = vass(states, controls, omega, obj, file)

Re = 6371000;
v = states(1);
chi = states(2);
gamma = states(3);
teta = states(4);
lam = states(5);
h = states(6);
m = states(7);
vv = [-v * cos(gamma) * cos(chi), v * cos(gamma) * sin(chi), -v * sin(gamma)];
vv(1) = vv(1) + omega * cos(lam) * (Re + h);

if vv(1) <= 0.0
    if abs(vv(1)) >= abs(vv(2))
        chiass = atan(abs(vv(2) / vv(1)));
        if vv(2) < 0.0
            chiass = -chiass;
        end
    end
    if abs(vv(1)) < abs(vv(2))
        chiass = pi*0.5 - atan(abs(vv(1) / vv(2)));
        if vv(2) < 0.0
            chiass = -chiass;
        end
    end
end
if vv(1) > 0.0
    if abs(vv(1)) >= abs(vv(2))
        chiass = pi - atan((abs(vv(2)/vv(1))));
        if vv(2) < 0.0
            chiass = - chiass;
        end
    end
    if abs(vv(1)) < abs(vv(2))
        chiass = pi * 0.5 + atan(abs(vv(1) / vv(2)));
        if vv(2) < 0.0
            chiass = -chiass;
        end
    end
end

x = [(Re + h) * cos(lam) * cos(teta), (Re + h) * cos(lam) * sin(teta), (Re + h) * sin(lam)];

dx = dynamicsVel(states, controls, obj, file);
xp = [dx(6) * cos(lam) * cos(teta) - (Re + h) * dx(5) * sin(lam) * cos(teta) - (Re + h) * dx(4) * cos(lam) * sin(teta), ...
      dx(6) * cos(lam) * sin(teta) - (Re + h) * dx(5) * sin(lam) * sin(teta) + (Re + h) * dx(4) * cos(lam) * cos(teta), ...
      dx(6) * sin(lam) + (Re + h) * dx(5) * cos(lam)];

dxp = [-omega * x(2), omega * x(1), 0.0];

vtot = [xp(1) + dxp(1), xp(2) + dxp(2), xp(3) + dxp(3)];

vela = sqrt(vtot(1) ^ 2 + vtot(2) ^ 2 + vtot(3) ^ 2);

end


  