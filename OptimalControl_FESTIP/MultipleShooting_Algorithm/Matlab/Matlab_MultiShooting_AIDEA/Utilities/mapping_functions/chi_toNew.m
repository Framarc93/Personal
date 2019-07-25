function chinew = chi_toNew(value)
% input value must be ini radians
chinew = to_new_int(value, deg2rad(90.0), deg2rad(270.0), 0.0, 1.0);
end

