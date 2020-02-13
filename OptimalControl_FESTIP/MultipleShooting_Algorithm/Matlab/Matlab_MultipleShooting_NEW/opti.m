function [opt, fval] = opti(obj, prob, file, X0, LB, UB, states_init, cont_init)
global varOld costOld ineqOld eqOld


options=optimset('Algorithm', 'sqp', ...                         % declare algorithm to use, here set to sequential quadratic programming (SQP)
    'Display', 'iter-detailed', ...                                         %  displays output at each iteration, and gives the technical exit message.
    'TolCon', 1e-7, ...                                                     % tolerance on the constraint violation %'TolX', 1e-9, ...         % tolerance on the design vector     %
    'TolF', 1e-8, ...  
    'MaxSQPIter',1e5, ...                                                      % maximum number of SQP iterations allowed
    'MaxFunEvals',1e5, ...                                      % maximum number of function evaluations allowed
    'ScaleProblem','None', ...                                    %  causes the algorithm to normalize all constraints and the objective function    
    'UseParallel',true, ...
    'TolX',1e-16...
    );  %     'TolFun', 1e-6, ...   

[opt, fval] = fmincon(@(x)cost_fun(x, prob, obj, file, states_init, cont_init), ...
    X0, [], [], [], [], ...
    LB, UB, ...
    @(x)constraints(x, prob, obj, file, states_init, cont_init), options);


function [ineq, eq] = constraints(var, prob, obj, file, states_init, cont_init) 

if isequal(var, varOld)
        ineq = ineqOld;
        eq = eqOld;
else
    [ineq_c, eq_c, cost] = MultipleShooting(var, prob, obj, file, states_init, cont_init);
    ineq = ineq_c;
    eq = eq_c;
    

end
end

function out = cost_fun(var, prob, obj, file, states_init, cont_init)
global dataS
if isequal(var, varOld)
    out = costOld;
else
    [ineq, eq, cost] = MultipleShooting(var, prob, obj, file, states_init, cont_init);
    out = cost;
    

end
end

end




