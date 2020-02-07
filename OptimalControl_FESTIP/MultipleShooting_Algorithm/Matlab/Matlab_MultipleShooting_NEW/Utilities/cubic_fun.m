function [outputArg1,outputArg2] = cubic_fun(inputArg1,inputArg2)
%CUBIC_FUN Summary of this function goes here
%   Detailed explanation goes here
outputArg1 = inputArg1;
outputArg2 = inputArg2;
end

""" return cubic function values that array size is same as time length
        Args:
            time (array_like) : time
            y0 (float) : initial value
            yprime0 (float) : slope of initial value
            yf (float) : final value
            yprimef (float) : slope of final value
        Returns:
            (N, ) ndarray
        """
        y = np.array([y0, yprime0, yf, yprimef])
        t0 = time[0]
        tf = time[-1]
        A = np.array([[1, t0, t0**2, t0**3], [0, 1, 2*t0, 3*t0**2],
                      [1, tf, tf**2, tf**3], [0, 1, 2*tf, 3*tf**2]])
        invA = np.linalg.inv(A)
        C = invA.dot(y)
        ys = C[0] + C[1]*time + C[2]*time**2 + C[3]*time**3