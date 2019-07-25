for iRun=1:20
    disp(iRun)
    load('bSol.mat')
%X0 = (bSol.xd - prob.LBV)./(prob.UBV-prob.LBV);
X0=bSol.x
out = Plot_MS(X0, prob, obj, file)

% XVminl=max([X0-0.001; zeros(size(X0))]);
% XVmaxl=min([X0+0.001; ones(size(X0))]);
%             
% X0=rand(size(X0)).*repmat(XVmaxl-XVminl,1,1)+repmat(XVminl,1,1);
options=optimset('Algorithm', 'sqp', ...                         % declare algorithm to use, here set to sequential quadratic programming (SQP)
    'Display', 'iter-detailed', ...                                         %  displays output at each iteration, and gives the technical exit message.
    'TolCon', 1e-6, ...                                                     % tolerance on the constraint violation %'TolX', 1e-9, ...         % tolerance on the design vector     %
    'TolF', 1e-4, ...  
    'MaxSQPIter',2300, ...                                                      % maximum number of SQP iterations allowed
    'MaxFunEvals',1e5, ...                                      % maximum number of function evaluations allowed
    'ScaleProblem','none', ...                                    %  causes the algorithm to normalize all constraints and the objective function    
    'UseParallel',false, ...
    'TolX',1e-15...
    );  %     'TolFun', 1e-6, ...   
[opt, fval] = fmincon(@(x)mask_MS(x, obj, prob, file), ...
    X0, [], [], [], [], ...
    LB, UB, ...
    [], options);
end