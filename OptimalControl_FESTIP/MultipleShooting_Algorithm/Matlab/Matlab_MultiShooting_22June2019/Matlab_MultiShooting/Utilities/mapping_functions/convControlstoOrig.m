function new_var = convControlstoOrig(var, Ncontrols)
    new_var = zeros(1, Ncontrols);
    new_var(1) = alfa_toOrig(var(1));
    new_var(2) = var(2);
    new_var(3) = deltaf_toOrig(var(3));
    new_var(4) = tau_toOrig(var(4));
    new_var(5) = mu_toOrig(var(5));
end

