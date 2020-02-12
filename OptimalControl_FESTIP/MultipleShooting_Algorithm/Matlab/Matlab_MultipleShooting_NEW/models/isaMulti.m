function [pressure, density, csound] = isaMulti(altitude, obj)

p0 = obj.psl;
t0 = 288.15;
prevh = 0.0;
g0 = obj.g0;
R = 287.00;
m0 = 28.9644;
Rs = 8314.32;
r = obj.Re;
temperature = [];
pressure = [];
tempm = [];
density = [];
csound = [];
zbf = obj.h90(end-1);
zf = obj.h90(end);
bf = zbf - obj.tcoeff1(end) / obj.a90(end);
tempf = obj.tcoeff1(end) + (obj.tcoeff2(end) * (zf - zbf)) / 1000;
tmf = obj.tmcoeff(end) + obj.a90(end) * (zf - zbf) / 1000;
add1f = 1 / ((r + bf) * (r + zf)) + (1 / ((r + bf) ^ 2)) * log(abs((zf - bf) / (zf + r)));
add2f = 1 / ((r + bf) * (r + zbf)) + (1 / ((r + bf) ^ 2)) * log(abs((zbf - bf) / (zbf + r)));
pressf = obj.pcoeff(end) * exp(-m0 / (obj.a90(end) * Rs) * g0 * r ^ 2 * (add1f - add2f));
densf = pressf / (R * tempf);
sspeedf = sqrt(1.4 * R * tmf);

for i=1:length(altitude)
    alt = altitude(i);
    if alt < 0 || isnan(alt)
        t = t0;
        p = p0;
        d = p / (R * t);
        c = sqrt(1.4 * R * t);
        density = [density, d];
        csound = [csound, c];
        pressure = [pressure, p];
    else
        k=1;
        if  (ge(alt,0) && alt < 90000)
            while (k<=8)
                if (le(alt,obj.hv(k)))
                    [temp,press]=cal(p0,t0,obj.a(k),prevh,alt, R, g0);
                    dens = press / (R * temp);
                    sspeed = sqrt(1.4 * R * temp);
                    density = [density, dens];
                    csound = [csound, sspeed];
                    pressure = [pressure, press];
                    t0 = 288.15;
                    p0 = obj.psl;
                    prevh = 0;
                    break
                else
                    [t0, p0] = cal(p0, t0, obj.a(k), prevh, obj.hv(k), R, g0);
                    prevh = obj.hv(k);
                end
                k=k+1;
            end
        else
            if (ge(alt,90000) && le(alt,190000))
                [temp, press, tpm] = atm90(obj.a90, alt, obj.h90, obj.tcoeff1, obj.pcoeff, obj.tcoeff2, obj.tmcoeff, m0, Rs, r, g0);
                dens = press / (R * temp);
                sspeed = sqrt(1.4 * R * tpm);
                density = [density, dens];
                csound = [csound, sspeed];
                pressure = [pressure, press];
            else
                if (alt > 190000) || isinf(alt)
                    density = [density, densf];
                    csound = [csound, sspeedf];
                    pressure = [pressure, pressf];
                end
            end
        end
    end
    if i~= size(pressure)
        disp("errore")
    end
end

    function [t1, p1] = cal(ps, ts, av, h0, h1, R, g0)
       
        if (av ~= 0)
            t1 = ts + av * (h1 - h0);
            p1 = ps * (t1 / ts) ^ (-g0 / av / R);
        else
            t1 = ts;
            p1 = ps * exp(-g0 / R / ts * (h1 - h0));
        end
    end

    function [temp,press,tm] = atm90(a90v, z ,hi, tc1, pc, tc2, tmc, m0, Rs, r, g0)
       
        ni=1;
        while (le(ni,length(hi)))
            if le(z,hi(ni))
                ind=ni;
                if (ind == 1)
                    zb = hi(1);
                    b = zb - tc1(1) / a90v(1);
                    temp = tc1(1) + tc2(1) * (z - zb) / 1000;
                    tm = tmc(1) + a90v(1) * (z - zb) / 1000;
                    add1 = 1 / ((r + b) * (r + z)) + (1 / ((r + b)^ 2)) * log(abs((z - b) / (z + r)));
                    add2 = 1 / ((r + b) * (r + zb)) + (1 / ((r + b)^ 2)) * log(abs((zb - b) / (zb + r)));
                    press = pc(1) * exp(-m0 / (a90v(1) * Rs) * g0 * r ^ 2 * (add1 - add2));
                else
                    zb = hi(ind-1);
                    b = zb - tc1(ind-1) / a90v(ind-1);
                    temp = tc1(ind-1) + (tc2(ind-1) * (z - zb)) / 1000;
                    tm = tmc(ind - 1) + a90v(ind - 1) * (z - zb) / 1000;
                    add1 = 1 / ((r + b) * (r + z)) + (1 / ((r + b) ^ 2)) * log(abs((z - b) / (z + r)));
                    add2 = 1 / ((r + b) * (r + zb)) + (1 / ((r + b) ^ 2)) * log(abs((zb - b) / (zb + r)));
                    press = pc(ind - 1) * exp(-m0 / (a90v(ind - 1) * Rs) * g0 * r ^ 2 * (add1 - add2));
                    break
                end
            end
            ni=ni+1;
        end
    end

end