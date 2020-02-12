function [press, dens, sspeed] = isa_FESTIP(altitude, obj)
    
% if isnan(altitude)
%     altitude = 1;
% else
%     if isinf(altitude)
%         altitude = obj.hmax;
%     end
% end

p0 = obj.psl;
t0 = 288.15;
prevh = 0.0;
g0 = obj.g0;
R = 287.00;
m0 = 28.9644;
Rs = 8314.32;
r = obj.Re;

if altitude < 0 || isnan(altitude)
    temp = t0;
    press = p0;
    dens = press / (R * temp);
    sspeed = sqrt(1.4 * R * temp);
end
if (ge(altitude,0) && altitude < 90000) 
    for i=1:8
        if (le(altitude,obj.hv(i)))
            [temp,press]=cal(p0,t0,obj.a(i),prevh,altitude, g0, R);
            dens = press / (R * temp);
            sspeed = sqrt(1.4 * R * temp);
            break
        else
            [t0, p0] = cal(p0, t0, obj.a(i), prevh, obj.hv(i), g0, R);
            prevh = obj.hv(i);
        end
    end
end

if (ge(altitude,90000) && le(altitude,190000))
        [temp, press, tpm] = atm90(obj.a90, altitude, obj.h90, obj.tcoeff1, obj.pcoeff, obj.tcoeff2, obj.tmcoeff, g0, r, Rs, m0);
        dens = press / (R * temp);
        sspeed = sqrt(1.4 * R * tpm);
end

if (altitude > 190000) || isinf(altitude)
   zb = obj.h90(7);
   z = obj.h90(end);
   b = zb - obj.tcoeff1(7) / obj.a90(7);
   temp = obj.tcoeff1(7) + (obj.tcoeff2(7) * (z - zb)) / 1000;
   tm = obj.tmcoeff(7) + a90(7) * (z - zb) / 1000;
   add1 = 1 / ((r + b) * (r + z)) + (1 / ((r + b) ^ 2)) * log(abs((z - b) / (z + r)));
   add2 = 1 / ((r + b) * (r + zb)) + (1 / ((r + b) ^ 2)) * log(abs((zb - b) / (zb + r)));
   press = obj.pcoeff(7) * exp(-m0 / (a90(7) * Rs) * g0 * r ^ 2 * (add1 - add2));
   dens = press / (R * temp);
   sspeed = sqrt(1.4 * R * tm);
end
    

 function [t1, p1] = cal(ps, ts, av, h0, h1, g0, R)
    if (av ~= 0)
        t1 = ts + av * (h1 - h0);
        p1 = ps * (t1 / ts) ^ (-g0 / av / R);
    else
        t1 = ts;
        p1 = ps * exp(-g0 / R / ts * (h1 - h0));
    end
end
    
function [temp,press,tm] = atm90(a90v, z ,hi, tc1, pc, tc2, tmc, g0, r, Rs, m0)
    ni=1;
    while (le(ni,length(hi)))
        if le(z,hi(ni))
            ind=ni;
            if (ind == 1)
                zb90 = hi(1);
                b90 = zb90 - tc1(1) / a90v(1);
                temp = tc1(1) + tc2(1) * (z - zb90) / 1000;
                tm = tmc(1) + a90v(1) * (z - zb90) / 1000;
                add1_90 = 1 / ((r + b90) * (r + z)) + (1 / ((r + b90)^ 2)) * log(abs((z - b90) / (z + r)));
                add2_90 = 1 / ((r + b90) * (r + zb90)) + (1 / ((r + b90)^ 2)) * log(abs((zb90 - b90) / (zb90 + r)));
                press = pc(1) * exp(-m0 / (a90v(1) * Rs) * g0 * r ^ 2 * (add1_90 - add2_90));
             else
                zb90 = hi(ind-1);
                b90 = zb90 - tc1(ind-1) / a90v(ind-1);
                temp = tc1(ind-1) + (tc2(ind-1) * (z - zb90)) / 1000;
                tm = tmc(ind - 1) + a90v(ind - 1) * (z - zb90) / 1000;
                add1_90 = 1 / ((r + b90) * (r + z)) + (1 / ((r + b90) ^ 2)) * log(abs((z - b90) / (z + r)));
                add2_90 = 1 / ((r + b90) * (r + zb90)) + (1 / ((r + b90) ^ 2)) * log(abs((zb90 - b90) / (zb90 + r)));
                press = pc(ind - 1) * exp(-m0 / (a90v(ind - 1) * Rs) * g0 * r ^ 2 * (add1_90 - add2_90));
                break
             end
        end
        ni=ni+1;
    end
 end    
end

