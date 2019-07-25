function [opt, fval] = opti(obj, prob, file, X0, LB, UB)
global varOld costOld eqOld ineqOld States Controls globTime



global bbb
bbb.f=1e16;
bbb.x=[];
bbb.xd=[];

opts = optimoptions(@patternsearch,'PlotFcn',{@psplotbestf,@psplotfuncount},'InitialMeshSize',0.5,'AccelerateMesh',true,'ScaleMesh',false,'Display','diagnose',...
                    'MeshExpansionFactor',1.6);

[X1,Fval,Exitflag,Output] = patternsearch(@(x)mask_MS_2(x, obj, prob, file), ...
    X0, [], [], [], [], ...
    LB, UB, ...
    [], opts);



options=optimset('Algorithm', 'interior-point', ...                         % declare algorithm to use, here set to sequential quadratic programming (SQP)
    'Display', 'iter-detailed', ...                                         %  displays output at each iteration, and gives the technical exit message.
    'TolCon', 1e-6, ...                                                     % tolerance on the constraint violation %'TolX', 1e-9, ...         % tolerance on the design vector     %
    'TolF', 1e-4, ...  
    'MaxSQPIter',2300, ...                                                      % maximum number of SQP iterations allowed
    'MaxFunEvals',1e5, ...                                      % maximum number of function evaluations allowed
    'ScaleProblem','none', ...                                    %  causes the algorithm to normalize all constraints and the objective function    
    'UseParallel',false, ...
    'TolX',1e-15...
    );  %     'TolFun', 1e-6, ...   
[opt, fval] = fmincon(@(x)cost_fun(x, obj, prob, file), ...
    X0, [], [], [], [], ...
    LB, UB, ...
    @(x)constraints(x, obj, prob, file), options);

function [ineq, eq] = constraints(var, obj, prob, file) 
global dataS
if isequal(var, varOld)
%         disp(' Constr R')
        eq = eqOld;
        ineq = ineqOld;
else
    tic
    [ineq_c, eq_c, ob] = MultipleShooting(var, obj, prob, file);
    tr=toc;
    eq = eq_c;
    ineq = ineq_c;
%     disp(' Constr ')
%     disp([tr ob])
%     disp('     ')
    dataS=[dataS; var ob max(ineq_c) max(abs(eq_c))];
    vv=dataS;
    save VVl vv
end
end

function out = cost_fun(var, obj, prob, file)
global dataS
if isequal(var, varOld)
    out = costOld;
else
    tic
    [ineq, eq, cost] = MultipleShooting(var, obj, prob, file);
    tr=toc;
    out = cost;
    dataS=[dataS; var cost max(ineq) max(abs(eq))];
    vv=dataS;
    save VVl vv
%     disp(' Obj ')
%     disp([tr out max(ineq) max(abs(eq))])  
%     TimeN = now;
%     d = datetime(TimeN,'ConvertFrom','datenum');
%     disp(d)
%     disp('     ')
end
end

end




