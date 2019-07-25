function [controls] = extract_controls(var, cp, nc)

controls = zeros(nc, cp);
z = 1;
for i=1:cp
    for j=1:nc
        controls(j,i) = var(z);
        z = z + 1;
    end
end

end

