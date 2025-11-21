function X_qbest = QOBL(X_best, Lb, Ub)
    % This function implements the Quasi-Opposition Based Learning (QOBL) approach
    % for the best solution in the I-GKSO algorithm
    
    % Inputs:
    % X_best - current best solution (optimal solution found so far)
    % Lb - lower bound of the search space
    % Ub - upper bound of the search space
    % c - parameter for the stochastic generation
    
    % Output:
    % X_qbest - quasi-opposite value of the best solution
    
    % Calculate the mirrored value of the optimal solution
    X_op_best = Lb + Ub - X_best;
    X_qbest = normrnd((Lb+Ub)/2,X_op_best);
    
    % Alternative simpler implementation if you want uniform distribution instead:
    % X_qbest = c + (X_op_best - c) * rand(); % uniform distribution
end