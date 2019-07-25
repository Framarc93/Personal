function [lbs, lbc, ubs, ubc] = bound_def(X, U, uplimx, inflimx, uplimu, inflimu)
sx = size(X);
su = size(U);
lbs = zeros(1, sx(2));
ubs = zeros(1, sx(2));
lbc = zeros(1, su(2));
ubc = zeros(1, su(2));
for i=1:sx(2)
    if X(i) == 0
        lbs(i) = inflimx(i) * 10 / 100;
        ubs(i) = uplimx(i) * 10 / 100;
    else
        lbs(i) = X(i) * (1 - 50 / 100);
        ubs(i) = X(i) * (1 + 50 / 100);
    end
    if lbs(i) < inflimx(i)
        lbs(i) = inflimx(i);
    end
    if ubs(i) > uplimx(i)
        ubs(i) = uplimx(i);
    end
end
for j=1:su(2)
    if U(j) == 0
        lbc(j) = inflimu(j) * 10 / 100;
        ubc(j) = uplimu(j) * 10 / 100;
    else
        lbc(j) = U(j) * (1 - 50 / 100);
        ubc(j) = U(j) * (1 + 50 / 100);
    end
    if lbc(j) < inflimu(j)
        lbc(j) = inflimu(j);
    end
    if ubc(j) > uplimu(j)
        ubc(j) = uplimu(j);
    end
end
end

