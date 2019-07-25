function [FT, deps, spimp, mommot] =  prop_FESTIP(presamb, m, presv, spimpv, delta, tau, obj)
presv = table2array(presv);
spimpv = table2array(spimpv);
slpres = obj.psl;
nimp = 17;
nmot = 1;
lref = obj.lRef;
wlo = obj.M0;
we = obj.m10;
xcgf = obj.xcgf;  %cg position with empty vehicle
xcg0 = obj.xcg0;  %cg position at take-off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

thrx = nmot*(5.8E+6+14.89*slpres-11.16*presamb)*delta;

if presamb >= slpres
    spimp = spimpv(end);
    presamb = slpres;
else
    i=1;
    if (presamb < obj.psl)
        while (i <= nimp)
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

xcg = ((xcgf  - xcg0) / (we-wlo) * (m - wlo) + xcg0) * lref;
dthr = 0.4224 * (36.656 - xcg) * thrx - 19.8 * (32 - xcg) * (1.7 * slpres - presamb);
mommot = tau * dthr;
thrz = -tau * (2.5E+6 - 22*slpres + 9.92 * presamb);
FT = sqrt(thrx^2+thrz^2);
deps = atan(thrz/thrx);





