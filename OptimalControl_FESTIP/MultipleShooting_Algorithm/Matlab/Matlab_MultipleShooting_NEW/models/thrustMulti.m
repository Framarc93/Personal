function [Thrust, Deps, Simp, Mom] =  thrustMulti(presamb, m, presv, spimpv, delta, tau, obj, npoint)

nimp = 17;
nmot = 1;
Deps = [];
Simp = [];
Mom = [];
Thrust = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j=1:npoint
    if m(j)<obj.m10 || isnan(m(j))
        m(j) = obj.m10;
    elseif m(j) > obj.M0 || isinf(m(j))
        m(j) = obj.M0;
    end
    thrx = nmot*(5.8e6 + 14.89*obj.psl-11.16*presamb(j))*delta(j);
    if presamb(j) > obj.psl
        spimp = spimpv(end);
        presamb(j) = obj.psl;
    else
        if (presamb(j) <= obj.psl)
            for i=1:nimp
                if i == nimp
                    spimp = spimpv(end) - (((presv(end)-presamb(j))/(presv(end)-presv(i-1)))*(spimpv(end)-spimpv(i-1)));
                    break
                else
                    if presv(i) >= presamb(j)
                        spimp = spimpv(i+1) - (((presv(i+1)-presamb(j))/(presv(i+1)-presv(i)))*(spimpv(i+1)-spimpv(i)));
                        break
                    end
                end
            end
        end
    end

    xcg = ((obj.xcgf  - obj.xcg0) / (obj.m10-obj.M0) * (m(j) - obj.M0) + obj.xcg0) * obj.lRef;
    dthr = 0.4224 * (36.656 - xcg) * thrx - 19.8 * (32 - xcg) * (1.7 * obj.psl - presamb(j));
    if tau(j) == 0
        mommot = 0;
        FT = thrx;
        deps = 0.0; 
    else
        mommot = tau(j) * dthr;
        thrz = -tau(j) * (2.5E+6 - 22*obj.psl + 9.92 * presamb(j));
        FT = sqrt(thrx^2+thrz^2);
        deps = atan(thrz/thrx);
    end
    Thrust = [Thrust, FT];
    Deps = [Deps, deps];
    Simp = [Simp, spimp];
    Mom = [Mom, mommot];
end
end





