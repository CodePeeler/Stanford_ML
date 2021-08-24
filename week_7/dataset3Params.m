function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

if false #set to true to find optimal values for C and sigma from paramVal.
    paramVal = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];
    t = length(paramVal);
    errors = zeros(t,t);
    
    for i=1:t
        C = paramVal(i);
        for j=1:t
            sigma = paramVal(j);
            #Train model with training data X, y.
            model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
            #Predictions using cross validation data.
            pred = svmPredict(model, Xval);
            #Calculate the error.
            errors(i,j) = mean(double(pred ~= yval));
        endfor
    endfor
    
    #visualize error matrix (just for fun!)
    surf(errors);
    fprintf('Program paused. Press enter to continue.\n');
    pause;
    
    #print error matrix.
    printf("\n------------- Error Matrix -------------------\n\n");
        errors
    printf("\n----------------------------------------------\n\n");
    
    #Find the coordinates of minimum error.
    [i,j] = find(errors==min(errors(:)));
    C = paramVal(i)
    sigma = paramVal(j)
else
    #These values were found by setting the above if=true.
    C = 1.0;
    sigma = 0.10;
endif
% =========================================================================


end 
