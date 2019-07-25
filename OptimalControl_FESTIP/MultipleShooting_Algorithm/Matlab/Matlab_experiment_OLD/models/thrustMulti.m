function [Thrust, Deps, Simp, Mom] =  thrustMulti(presamb, m, presv, spimpv, delta, tau, obj, npoint)

slpres = obj.psl;
nimp = 17;
nmot = 1;
lref = obj.lRef;
wlo = obj.M0;
we = obj.m10;
xcgf = obj.xcgf;  %cg position with empty vehicle
xcg0 = obj.xcg0;  %cg position at take-off
Deps = zeros(1, npoint);
Simp = zeros(1, npoint);
Mom = zeros(1, npoint);
Thrust = zeros(1, npoint);
presv = table2array(presv);
spimpv = table2array(spimpv);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j=1:npoint
    thrx = nmot*(5.8E+6+14.89*slpres-11.16*presamb(j))*delta(j);
    if presamb(j) > slpres
        spimp = spimpv(end);
        presamb(j) = slpres;
    else
        if (presamb(j) <= slpres)
            for i=1:nimp
                if i == nimp
                    spimp = spimpv(i) - (((presv(i)-presamb(j)/(presv(i)-presv(i-1)))*(spimpv(i)-spimpv(i-1))));
                    break
                else
                    if (ge(presv(i),presamb(j)))
                        spimp = spimpv(i+1) - (((presv(i+1)-presamb(j)/(presv(i+1)-presv(i)))*(spimpv(i+1)-spimpv(i))));
                        break
                    end
                end
            end
        end
    end

    xcg = ((xcgf  - xcg0) / (we-wlo) * (m(j) - wlo) + xcg0) * lref;
    dthr = 0.4224 * (36.656 - xcg) * thrx - 19.8 * (32 - xcg) * (1.7 * slpres - presamb(j));
    mommot = tau(j) * dthr;
    thrz = -tau(j) * (2.5E+6 - 22*slpres + 9.92 * presamb(j));
    FT = sqrt(thrx^2+thrz^2);
    deps = atan(thrz/thrx);
    Thrust(i) = FT;
    Deps(i) = deps;
    Simp(i) = spimp;
    Mom(i) = mommot;
end
end





