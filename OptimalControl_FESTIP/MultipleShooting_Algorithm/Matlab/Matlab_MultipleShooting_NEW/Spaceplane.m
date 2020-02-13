classdef Spaceplane
    properties
        GMe = 3.986004418e14;  % Earth gravitational constant [m^3/s^2]
        Re = 6371000;  % Earth Radius [m]
        psl = 101325;  % ambient pressure at sea level [Pa]
        latstart = deg2rad(5.2);  % deg latitude
        longstart = deg2rad(-52.775);  % deg longitude
        chistart = deg2rad(125);  % deg flight direction
        incl = deg2rad(51.6);  % deg orbit inclination
        gammastart = deg2rad(89);  % deg
        M0 = 450400;  % kg  starting mass
        g0 = 9.80665;  % m/s2
        omega = 7.2921159e-5;
        MaxQ = 40000;  % Pa
        MaxAx = 30;  % m/s2
        MaxAz = 15;  % m/s2
        Htarget = 400000;  % m target height after hohmann transfer
        wingSurf = 500.0;  % m2
        lRef = 34.0;  % m
        k = 5e3;  % [Nm] livello di precisione per trimmaggio
        xcgf = 0.37;  % cg position with empty vehicle
        xcg0 = 0.65;  % cg position at take-off
        pref = 21.25;
        Hini = 100000;
        m10=0;
        r2=0;
        tvert = 2 %[s] time for completely vertical take off
        Vtarget=0;
        chi_fin=0;
        Rtarget=0;
        States=[];
        Controls=[];
        vmax = 1e4;
        chimax = deg2rad(170);
        gammamax = deg2rad(89);
        tetamax = deg2rad(-10);
        lammax = deg2rad(30);
        hmax = 150000;
        vmin = 0.5;
        chimin = deg2rad(90);
        gammamin = deg2rad(-40);
        tetamin = deg2rad(-60);
        lammin = deg2rad(2);
        hmin = 0.5;
        alfamax = deg2rad(40);
        alfamin = deg2rad(-2);
        deltamax = 1.0;
        deltamin = 0.0;
        deltafmax = deg2rad(30);
        deltafmin = deg2rad(-20);
        taumax = 1;
        taumin = -1;
        mach = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0];
        angAttack = [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.5, 25.0, 30.0, 35.0, 40.0];
        bodyFlap = [-20, -10, 0, 10, 20, 30];
        a = [-0.0065, 0, 0.0010, 0.0028, 0, -0.0020, -0.0040, 0];
        a90 = [0.0030, 0.0050, 0.0100, 0.0200, 0.0150, 0.0100, 0.0070];
        hv = [11000, 20000, 32000, 47000, 52000, 61000, 79000, 90000];
        h90 = [90000, 100000, 110000, 120000, 150000, 160000, 170000, 190000];
        tmcoeff = [180.65, 210.65, 260.65, 360.65, 960.65, 1110.65, 1210.65];
        pcoeff = [0.16439, 0.030072, 0.0073526, 0.0025207, 0.505861E-3, 0.36918E-3, 0.27906E-3];
        tcoeff2 = [2.937, 4.698, 9.249, 18.11, 12.941, 8.12, 5.1];
        tcoeff1 = [180.65, 210.02, 257.0, 349.49, 892.79, 1022.2, 1103.4];
    end
    methods
        function self = Spaceplane(self)
            self.m10 = self.M0 * 0.1;
            self.r2 = self.Re + self.Htarget;
            self.Rtarget = self.Re + self.Hini;
            self.Vtarget = sqrt(self.GMe / self.Rtarget);  % m/s forse da modificare con velocita' assoluta
            self.chi_fin = 0.5 * pi + asin(cos(self.incl) / cos(self.latstart));
        end
    end
            
end

