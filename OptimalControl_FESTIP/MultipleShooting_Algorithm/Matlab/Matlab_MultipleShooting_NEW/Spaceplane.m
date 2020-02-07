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
        Vtarget=0;
        chi_fin=0;
        Rtarget=0;
        States=[];
        Controls=[];
        vmax = 1e4;
        chimax = deg2rad(150);
        gammamax = deg2rad(89.9);
        tetamax = deg2rad(-10);
        lammax = deg2rad(30);
        hmax = 150000;
        vmin = 0.5;
        chimin = deg2rad(90);
        gammamin = deg2rad(-60);
        tetamin = deg2rad(-60);
        lammin = deg2rad(2);
        hmin = 0.5;
        alfamax = deg2rad(40);
        alfamin = deg2rad(-2);
        deltamax = 1.0;
        deltamin = 0.0;
        
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

