function new_var = convStatestoOrig(var, Nstates, obj, leng)
new_var = zeros(1, Nstates*leng);
for i=1:leng
    
    new_var((i-1)*Nstates + 1) = v_toOrig(var((i-1)*Nstates + 1));
    new_var((i-1)*Nstates + 2) = chi_toOrig(var((i-1)*Nstates + 2));
    new_var((i-1)*Nstates + 3) = gamma_toOrig(var((i-1)*Nstates + 3));
    new_var((i-1)*Nstates + 4) = teta_toOrig(var((i-1)*Nstates + 4));
    new_var((i-1)*Nstates + 5) = lam_toOrig(var((i-1)*Nstates + 5), obj);
    new_var((i-1)*Nstates + 6) = h_toOrig(var((i-1)*Nstates + 6));
    new_var((i-1)*Nstates + 7) = m_toOrig(var((i-1)*Nstates + 7),obj);
end
end
