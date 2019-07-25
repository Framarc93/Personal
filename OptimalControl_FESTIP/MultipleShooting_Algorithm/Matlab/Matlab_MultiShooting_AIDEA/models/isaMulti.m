function [pressure, density, csound] = isaMulti(altitude, obj)

a = [-0.0065, 0, 0.0010, 0.0028, 0, -0.0020, -0.0040, 0];
a90 = [0.0030, 0.0050, 0.0100, 0.0200, 0.0150, 0.0100, 0.0070];
hv = [11000, 20000, 32000, 47000, 52000, 61000, 79000, 90000];
h90 = [90000, 100000, 110000, 120000, 150000, 160000, 170000, 190000];
tmcoeff = [180.65, 210.65, 260.65, 360.65, 960.65, 1110.65, 1210.65];
pcoeff = [0.16439, 0.030072, 0.0073526, 0.0025207, 0.505861E-3, 0.36918E-3, 0.27906E-3];
tcoeff2 = [2.937, 4.698, 9.249, 18.11, 12.941, 8.12, 5.1];
tcoeff1 = [180.65, 210.02, 257.0, 349.49, 892.79, 1022.2, 1103.4];
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
zbf = h90(end-1);
zf = h90(end);
bf = zbf - tcoeff1(end) / a90(end);
tempf = tcoeff1(end) + (tcoeff2(end) * (zf - zbf)) / 1000;
tmf = tmcoeff(end) + a90(end) * (zf - zbf) / 1000;
add1f = 1 / ((r + bf) * (r + zf)) + (1 / ((r + bf) ^ 2)) * log(abs((zf - bf) / (zf + r)));
add2f = 1 / ((r + bf) * (r + zbf)) + (1 / ((r + bf) ^ 2)) * log(abs((zbf - bf) / (zbf + r)));
pressf = pcoeff(end) * exp(-m0 / (a90(end) * Rs) * g0 * r ^ 2 * (add1f - add2f));
densf = pressf / (R * tempf);
sspeedf = sqrt(1.4 * R * tmf);

for i=1:size(altitude)
    
    alt = altitude(i);
    if isnan(alt)
        alt=100;
    end
    if alt < 0
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
                if (le(alt,hv(k)))
                    [temp,press]=cal(p0,t0,a(k),prevh,alt);
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
                    [t0, p0] = cal(p0, t0, a(k), prevh, hv(k));
                    prevh = hv(k);
                end
                k=k+1;
            end
        else
            if (ge(alt,90000) && le(alt,190000))
                [temp, press, tpm] = atm90(a90, alt, h90, tcoeff1, pcoeff, tcoeff2, tmcoeff);
                dens = press / (R * temp);
                sspeed = sqrt(1.4 * R * tpm);
                density = [density, dens];
                csound = [csound, sspeed];
                pressure = [pressure, press];
            else
                if (alt > 190000)
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

    function [t1, p1] = cal(ps, ts, av, h0, h1)
        
        t0 = 288.15;
        p0 = 101325;
        prevh = 0.0;
        R = 287.00;
        m0 = 28.9644;
        Rs = 8314.32;
        m0 = 28.9644;
        r = 6371000;
        g0 = 9.80665;
        
        if (av ~= 0)
            t1 = ts + av * (h1 - h0);
            p1 = ps * (t1 / ts) ^ (-g0 / av / R);
        else
            t1 = ts;
            p1 = ps * exp(-g0 / R / ts * (h1 - h0));
        end
    end

    function [temp,press,tm] = atm90(a90v, z ,hi, tc1, pc, tc2, tmc)
        t0 = 288.15;
        p0 = 101325;
        prevh = 0.0;
        R = 287.00;
        m0 = 28.9644;
        Rs = 8314.32;
        m0 = 28.9644;
        r = 6371000;
        g0 = 9.80665;
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