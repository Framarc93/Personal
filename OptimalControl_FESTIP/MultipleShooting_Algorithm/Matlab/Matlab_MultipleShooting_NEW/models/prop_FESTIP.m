function [FT, deps, spimp, mommot] =  prop_FESTIP(presamb, m, presv, spimpv, delta, tau, obj)
if m<obj.m10 || isnan(m)
        m = obj.m10;
    elseif m > obj.M0 || isinf(m)
        m = obj.M0;
    end
nimp = 17;
nmot = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

thrx = nmot*(5.8e6+14.89*obj.psl-11.16*presamb)*delta;

if presamb >= obj.psl
    spimp = spimpv(end);
    presamb = obj.psl;
else
    i=1;
    if presamb < obj.psl
        while i <= nimp
            if i == nimp
                spimp = spimpv(end) - (((presv(end)-presamb)/(presv(end)-presv(i-1)))*(spimpv(end)-spimpv(i-1)));
            else
                if presv(i) >= presamb
                    spimp = spimpv(i+1) - (((presv(i+1)-presamb)/(presv(i+1)-presv(i)))*(spimpv(i+1)-spimpv(i)));
                    break
                end
            end
            i = i + 1;
        end
    end
end

xcg = ((obj.xcgf  - obj.xcg0) / (obj.m10-obj.M0) * (m - obj.M0) + obj.xcg0) * obj.lRef;
dthr = 0.4224 * (36.656 - xcg) * thrx - 19.8 * (32 - xcg) * (1.7 * obj.psl - presamb);
if tau == 0
    mommot = 0.0;
    deps = 0.0;
    FT = thrx;
else
    mommot = tau * dthr;
    thrz = -tau * (2.5E+6 - 22*obj.psl + 9.92 * presamb);
    deps = atan(thrz/thrx);
    FT = sqrt(thrx^2+thrz^2);
end







