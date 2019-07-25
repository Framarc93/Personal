function res = to_new_int(t, a, b, c, d)
% this function converts a value from an interval [a, b] to [c, d]
% t = value to be converted
% a = inf lim old interval
% b = sup lim old interval
% c = inf lim new interval
% d = sup lim new interval
res = c + ((d-c)/(b-a))*(t-a);
end
