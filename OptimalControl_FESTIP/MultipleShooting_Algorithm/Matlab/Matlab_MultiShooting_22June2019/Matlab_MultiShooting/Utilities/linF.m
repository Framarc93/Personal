function y0=linF(x,y,x0)

y0 = (x0-x(1))./(x(2)-x(1)).*(y(2)-y(1))+y(1);