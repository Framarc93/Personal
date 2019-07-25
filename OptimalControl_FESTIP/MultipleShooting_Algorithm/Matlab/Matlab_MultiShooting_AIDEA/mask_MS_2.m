function [J]=mask_MS_2(var, obj, prob, file)
global bbb
try
[ineq_Cond, eq_Cond, objective] = MultipleShooting(var, obj, prob, file);

J = objective + sum(max([ineq_Cond'; zeros(size(ineq_Cond'))])) + sum(abs(eq_Cond));
if isnan(J) || ~isreal(J)
J=1e6+rand*1e6;
end
catch
J=1e8+rand*1e8;
end
if J<bbb.f
    bbb.f=J;
    bbb.x=var;
    varD = var.*(prob.UBV - prob.LBV) + prob.LBV;
    bbb.xd=varD;
    bSol=bbb;
    save bSol_PS bSol
end
