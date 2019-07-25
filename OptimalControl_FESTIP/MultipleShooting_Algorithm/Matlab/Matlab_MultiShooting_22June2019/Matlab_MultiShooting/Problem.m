classdef Problem
    properties
    %set problem parameters%
    Nbar = 5; % number of conjunction points
    NContPoints = 7;  % number of control points for interpolation inside each interval
    Nint = 150; % number of points for each single shooting integration
    Nstates = 7;  % number of states
    Ncontrols = 5;  % number of controls 
    NineqCond = 0;
    Nleg = 0;
    varStates = 0;
    varControls = 0;
    varTot = 0;
    varC = 0;
    %NLP solver parameters%
    maxiter = 1; % max number of iterations for nlp solver
    ftol = 1e-12;  % numeric tolerance of nlp solver
    %eps = 1e-10
    eps = 1.4901161193847656e-08; % increment of the derivative
    maxIterator = 1;  % max number of optimization iteration    
    unit_t = 1000;
    LBV = [];
    UBV = [];
    end
    methods
        function self = Problem(self)
            self.Nleg = self.Nbar - 1;  % number of multiple shooting sub intervals
            self.varStates = self.Nstates * self.Nleg; % total number of optimization variables for states
            self.varControls = self.Ncontrols * self.Nleg * self.NContPoints;   % total number of optimization variables for controls
            self.varTot = self.varStates + self.varControls; % total number of optimization variables for states and controls
            self.varC = round(self.varControls / self.Ncontrols);
            self.NineqCond = self.Nint;
        end
    end
end

