function [vela, chiass] = vass(states, controls, omega, obj, file)

Re = obj.Re;
v = states(1);
if isnan(v)
    v = 0.0;
elseif isinf(v) || v>1e6
    v = 1e6;
end

chi = states(2);
if isnan(chi)
    chi = 0.0;
elseif isinf(chi) || chi>obj.chimax
    chi = obj.chimax;
end
gamma = states(3);
if isnan(gamma)
    gamma = 0.0;
elseif isinf(gamma)
    gamma = 1e10;
end
teta = states(4);
lam = states(5);
if isnan(lam)
    lam = 0.0;
elseif isinf(lam)
    lam = 1e10;
end
h = states(6);
if isnan(h)
    h = 0.0;
elseif isinf(h) || h>1e7
    h = 1e7;
end
m = states(7);
if isnan(m) || m <obj.m10
    m=obj.m10;
elseif isinf(m) || m> obj.M0
    m=obj.M0;
end

vv = [-v * cos(gamma) * cos(chi), v * cos(gamma) * sin(chi), -v * sin(gamma)];
vv(1) = vv(1) + omega * cos(lam) * (Re + h);
vela2 = sqrt(vv(1)^2 + vv(2)^2 + vv(3)^2);
if vv(1) <= 0.0 || isnan(vv(1))
    if abs(vv(1)) >= abs(vv(2))
        chiass = atan(abs(vv(2) / vv(1)));
        if vv(2) < 0.0 || isnan(vv(2))
            chiass = -chiass;
        end
    end
    if abs(vv(1)) < abs(vv(2))
        chiass = pi*0.5 - atan(abs(vv(1) / vv(2)));
        if vv(2) < 0.0 || isnan(vv(2))
            chiass = -chiass;
        end
    end
end
if vv(1) > 0.0 || isinf(vv(1))
    if abs(vv(1)) >= abs(vv(2))
        chiass = pi - atan((abs(vv(2)/vv(1))));
        if vv(2) < 0.0 || isnan(vv(2))
            chiass = - chiass;
        end
    end
    if abs(vv(1)) < abs(vv(2))
        chiass = pi * 0.5 + atan(abs(vv(1) / vv(2)));
        if vv(2) < 0.0 || isnan(vv(2))
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


  