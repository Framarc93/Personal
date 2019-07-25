function res = linear_fun(time, y0, yf)
    x = [time(1), time(end)];
    y = [y0, yf];
    f = interp1(x, y);
    res = f(time);
end
