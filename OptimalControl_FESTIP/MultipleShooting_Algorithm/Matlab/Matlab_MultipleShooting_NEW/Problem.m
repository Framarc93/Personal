classdef Problem
    properties
    %set problem parameters%
    NContPoints = 5;  % number of control points for interpolation inside each interval
    NContPointsLeg1 = 4;
    Nint = 100; % number of points for each single shooting integration
    discretization = 1; % [s] how close are to each other the propagation points
    Nstates = 7;  % number of states
    Ncontrols = 4;  % number of controls 
    NineqCond = 5;
    varStates = 0;
    varControls = 0;
    varTot = 0;
    varC = 0;
    Nleg = 6;
    Nbar = 0;
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
            self.Nbar = self.Nleg + 1;
            self.varStates = self.Nstates * (self.Nleg-1) + 1;
            self.varControls = self.Ncontrols * (self.NContPoints * (self.Nleg-1) + self.NContPointsLeg1);   % total number of optimization variables for controls
            self.varTot = self.varStates + self.varControls; % total number of optimization variables for states and controls
            self.varC = round(self.varControls / self.Ncontrols);
        end
    end
end

