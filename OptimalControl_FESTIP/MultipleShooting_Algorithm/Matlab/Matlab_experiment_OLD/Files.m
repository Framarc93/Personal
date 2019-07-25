classdef Files
   
    properties
        cl = importfile("/home/francesco/git_workspace/FESTIP_Work/coeff_files/clfile.txt", "w");
        cd = importfile("/home/francesco/git_workspace/FESTIP_Work/coeff_files/cdfile.txt", "w");
        cm = importfile("/home/francesco/git_workspace/FESTIP_Work/coeff_files/cmfile.txt", "w");
        impulse = import_impulse("/home/francesco/git_workspace/FESTIP_Work/coeff_files/impulse.dat", "w");
        
        presv = 0;
        spimpv = 0;
    end
    methods
        function self = Files(self)
            self.presv = self.impulse(:, 1); 
            self.spimpv = self.impulse(:, 2);
            self.cl = table2array(self.cl);
            self.cd = table2array(self.cd);
            self.cm = table2array(self.cm);
            %self.cl = self.cl{:,:};
            %self.cl = reshape(self.cl, [17, 6, 13]);
            %self.cd = self.cd{:,:};
            %self.cd = reshape(self.cd, [17, 6, 13]);
            %self.cm = self.cm{:,:};
            %self.cm = reshape(self.cm, [17, 6, 13]);
        end
    end
   
end

