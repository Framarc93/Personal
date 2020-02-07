classdef Files
   
    properties
        cl = table2array(importfile("clfile.txt", "w"));
        cd = table2array(importfile("cdfile.txt", "w"));
        cm = table2array(importfile("cmfile.txt", "w"));
        impulse = table2array(import_impulse("impulse.dat", "w"));
        
        presv = 0;
        spimpv = 0;
    end
    methods
        function self = Files(self)
            self.presv = self.impulse(:, 1); 
            self.spimpv = self.impulse(:, 2);
            %self.cl = self.cl{:,:};
            %self.cl = reshape(self.cl, [17, 6, 13]);
            %self.cd = self.cd{:,:};
            %self.cd = reshape(self.cd, [17, 6, 13]);
            %self.cm = self.cm{:,:};
            %self.cm = reshape(self.cm, [17, 6, 13]);
        end
    end
   
end

