function [opt, fval] = opti(obj, prob, file, X0, LB, UB)
global varOld costOld eqOld ineqOld 
options=optimset('Algorithm', 'sqp', ...                         % declare algorithm to use, here set to sequential quadratic programming (SQP)
    'Display', 'iter-detailed', ...                                         %  displays output at each iteration, and gives the technical exit message.
    'TolCon', 1e-4, ...                                                     % tolerance on the constraint violation %'TolX', 1e-9, ...         % tolerance on the design vector     %
    'MaxSQPIter',20, ...                                                      % maximum number of SQP iterations allowed
    'MaxFunEvals',2000, ...                                      % maximum number of function evaluations allowed
    'ScaleProblem','none', ...                                    %  causes the algorithm to normalize all constraints and the objective function    
    'UseParallel',false...
    );  %     'TolFun', 1e-6, ...   
   
[opt, fval] = fmincon(@(x)cost_fun(x, obj, prob, file), ...
    X0, [], [], [], [], ...
    LB, UB, ...
    @(x)constraints(x, obj, prob, file), options);

function [ineq, eq] = constraints(var, obj, prob, file) 

if isequal(var, varOld)
        eq = eqOld;
        ineq = ineqOld;
else
    [ineq_c, eq_c, ob] = MultipleShooting(var, obj, prob, file);
    eq = eq_c;
    ineq = ineq_c;
end
end

function out = cost_fun(var, obj, prob, file)

if isequal(var, varOld)
    out = costOld;
else
    [ineq, eq, cost] = MultipleShooting(var, obj, prob, file);
    out = cost;
end
end

end




