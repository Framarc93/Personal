function [L,D, Mom] =  aeroForcesMulti(M, alfa, deltaf, cd, cl, cm, v, rho, mass, obj, npoint)

        
mach = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0];
angAttack = [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.5, 25.0, 30.0, 35.0, 40.0];
bodyFlap = [-20, -10, 0, 10, 20, 30];
lRef = obj.lRef;
mstart = obj.M0;
m10 = obj.m10;
xcgf = obj.xcgf; %cg position with empty vehicle
xcg0 = obj.xcg0; %cg position at take-off
pref = obj.pref;
sup = obj.wingSurf;
alfag = rad2deg(alfa);
deltafg = rad2deg(deltaf);
L = [];
D = [];
Mom = [];
for i=1:npoint
    cL = coefCalc(cl, M(i), alfag(i), deltafg(i)); 
    cD = coefCalc(cd, M(i), alfag(i), deltafg(i)); 
    l = 0.5 * (v(i) ^ 2) * sup * rho(i) * cL;
    d = 0.5 * (v(i) ^ 2) * sup * rho(i) * cD;
    xcg = lRef * (((xcgf - xcg0)/(m10 - mstart))*(mass(i)-mstart)+xcg0);
    dx = xcg - pref;
    cM = coefCalc(cm, M(i), alfag(i), deltafg(i));
    cM = cM + cL * (dx/lRef) * cos(alfa(i)) + cD * (dx/lRef) * sin(alfa(i));
    mom = 0.5 * (v(i) ^ 2) * sup * lRef * rho(i) * cM;
    L = [L, l];
    D = [D, d];
    Mom = [Mom, mom];
end

function coeffFinal = coefCalc(coeff, m, alfa, deltaf)
        if isnan(m)
            m=mach(end);
        end
        if m > mach(end)
            m = mach(end);
        else
            if m < mach(1)
                m = mach(1);
            end
        end
        if alfa > angAttack(end)
            alfa = angAttack(end);
        else
            if alfa < angAttack(1)
                alfa = angAttack(1);
            end
        end
        if deltaf > bodyFlap(end)
            deltaf = bodyFlap(end);
        else
            if deltaf < bodyFlap(1)
                deltaf = bodyFlap(1);
            end
        end
try        
        [im,sm] = limCalc(mach, m); %moments boundaries and determination of the 2 needed tables
        
        cnew1 = coeff(17*(im-1)+1:17*(im-1)+length(angAttack),:);
        cnew2 = coeff(17*(sm-1)+1:17*(sm-1)+length(angAttack),:);

        [ia,sa] = limCalc(angAttack, alfa); %angle of attack boundaries
        [id,sd] = limCalc(bodyFlap, deltaf); %PROBLEMA %deflection angle boundaries

        rowinf1 = cnew1(ia,:);
        rowsup1 = cnew1(sa,:);

        coeffinf1 = [rowinf1(id), rowsup1(id)]; %#colonna inferiore
        coeffsup1 = [rowinf1(sd), rowsup1(sd)]; %#colonna superiore

        c11 = linF([angAttack(ia), angAttack(sa)], coeffinf1, alfa); %interpolazione 1D
        c21 = linF([angAttack(ia), angAttack(sa)], coeffsup1, alfa); %interpolazione 1D
catch
    disp('ERROR coefCalc')
end
        coeffd1 = linF([bodyFlap(id), bodyFlap(sd)], [c11, c21],deltaf); %?

        %%%%%%%%%%%   2    %%%%%%%%%%

        rowinf2 = (cnew2(ia,:));
        rowsup2 = (cnew2(sa,:));

        coeffinf2 = [rowinf2(id), rowsup2(id)];
        coeffsup2 = [rowinf2(sd), rowsup2(sd)];

        c12 = linF([angAttack(ia), angAttack(sa)], coeffinf2, alfa); %interpolazione 1D
        c22 = linF([angAttack(ia), angAttack(sa)], coeffsup2, alfa); %interpolazione 1D
        coeffd2 = linF([bodyFlap(id), bodyFlap(sd)], [c12, c22],deltaf);

        %%%%%%%%%%%%%%%

        coeffFinal = linF([mach(im), mach(sm)],[coeffd1, coeffd2],m);
    end
    
    function [i,s] = limCalc(array, value)
    s = min(find(array>=value));
    s = max([s, 2]);
    i = max(find(array(1:s-1)<=value));
    
     
    end
end    