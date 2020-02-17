function [L,D, Mom] =  aero_FESTIP(M, alfa, deltaf, cd, cl, cm, v, rho, mass, obj)

if isnan(v)
    v = 0;
elseif isinf(v) || v>1e6
    v = 1e6;
end

alfag = rad2deg(alfa);
deltafg = rad2deg(deltaf);
%[X1] = ndgrid(angAttack,bodyFlap, mach);
%cL = interpn(X1, cl, alfag, deltafg, M)

cL = coefCalc(cl, M, alfag, deltafg, obj.mach, obj.angAttack, obj.bodyFlap); 
cD = coefCalc(cd, M, alfag, deltafg, obj.mach, obj.angAttack, obj.bodyFlap); 

L = 0.5 * (v ^ 2) * obj.wingSurf * rho * cL;
D = 0.5 * (v ^ 2) * obj.wingSurf * rho * cD;
xcg = obj.lRef * (((obj.xcgf - obj.xcg0)/(obj.m10 - obj.M0))*(mass-obj.M0)+obj.xcg0);
dx = xcg - obj.pref;
cM = coefCalc(cm, M, alfag, deltafg, obj.mach, obj.angAttack, obj.bodyFlap);
cM = cM + cL * (dx/obj.lRef) * cos(alfa) + cD * (dx/obj.lRef) * sin(alfa);
Mom = 0.5 * (v ^ 2) * obj.wingSurf * obj.lRef * rho * cM;
% catch
%     disp('ERROR ext')
% end

function coeffFinal = coefCalc(coeff, m, alfa, deltaf, mach, angAttack, bodyFlap)

    if m > mach(end) || isinf(m) 
        m = mach(end);
    elseif m < mach(1) || isnan(m)
            m = mach(1);
    end
    if alfa > angAttack(end) || isinf(alfa)
        alfa = angAttack(end);
    elseif alfa < angAttack(1) || isnan(alfa)
            alfa = angAttack(1);
    end
    if deltaf > bodyFlap(end) || isinf(deltaf)
        deltaf = bodyFlap(end);
    elseif deltaf < bodyFlap(1) || isnan(deltaf)
            deltaf = bodyFlap(1);
    end

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
    rowinf1 = (cnew1(ia,:));
    rowsup1 = (cnew1(sa,:));

    coeffinf1 = [rowinf1(id), rowsup1(id)]; %#colonna inferiore
    coeffsup1 = [rowinf1(sd), rowsup1(sd)]; %#colonna superiore

    c11 = linF([angAttack(ia), angAttack(sa)], coeffinf1, alfa); %interpolazione 1D
    c21 = linF([angAttack(ia), angAttack(sa)], coeffsup1, alfa); %interpolazione 1D

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
s = find(array>=value, 1);
s = max([s, 2]);
i = find(array(1:s-1)<=value, 1, 'last' );
end

end