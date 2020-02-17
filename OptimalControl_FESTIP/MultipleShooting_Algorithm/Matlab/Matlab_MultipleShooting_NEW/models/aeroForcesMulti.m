function [L,D, Mom] =  aeroForcesMulti(M, alfa, deltaf, cd, cl, cm, v, rho, mass, obj, npoint)

alfag = rad2deg(alfa);
deltafg = rad2deg(deltaf);
L = [];
D = [];
Mom = [];
for i=1:npoint
    if v(i) > 1e6 || isinf(v(i))
        v(i) = 1e6;
    elseif isnan(v(i))
        v(i) = 0;
    end
    cL = coefCalc(cl, M(i), alfag(i), deltafg(i), obj.mach, obj.angAttack, obj.bodyFlap); 
    cD = coefCalc(cd, M(i), alfag(i), deltafg(i), obj.mach, obj.angAttack, obj.bodyFlap); 
    l = 0.5 * (v(i) ^ 2) * obj.wingSurf * rho(i) * cL;
    d = 0.5 * (v(i) ^ 2) * obj.wingSurf * rho(i) * cD;
    xcg = obj.lRef * (((obj.xcgf - obj.xcg0)/(obj.m10 - obj.M0))*(mass(i)-obj.M0)+obj.xcg0);
    dx = xcg - obj.pref;
    cM = coefCalc(cm, M(i), alfag(i), deltafg(i), obj.mach, obj.angAttack, obj.bodyFlap);
    cM = cM + cL * (dx/obj.lRef) * cos(alfa(i)) + cD * (dx/obj.lRef) * sin(alfa(i));
    mom = 0.5 * (v(i) ^ 2) * obj.wingSurf * obj.lRef * rho(i) * cM;
    L = [L, l];
    D = [D, d];
    Mom = [Mom, mom];
end

function coeffFinal = coefCalc(coeff, m, alfa, deltaf, mach, angAttack, bodyFlap)
        if m > mach(end) || isinf(m) 
            m = mach(end);
        else
            if m < mach(1) || isnan(m)
                m = mach(1);
            end
        end
        if alfa > angAttack(end) || isinf(alfa)
            alfa = angAttack(end); 
        else
            if alfa < angAttack(1) || isnan(alfa)
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
        if deltaf == 0
            id = 2;
            sd = 3;
        else
            [id,sd] = limCalc(bodyFlap, deltaf); %PROBLEMA %deflection angle boundaries
        end
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
    s = find(array>=value, 1 );
    s = max([s, 2]);
    i = find(array(1:s-1)<=value, 1, 'last' );
    
     
    end
end    