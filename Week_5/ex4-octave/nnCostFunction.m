function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
% 
% ===================== MY CODE P1 -> START ======================
 # Add bias units.
 X = [ones(m,1), X];
 Z_2 = X*Theta1';
 # Add bias units.
 A_2 = [ones(m,1), sigmoid(Z_2)];
 Z_3 = A_2*Theta2';
 # Each column of A_3 represents a class, and each row represents the result of 
 # an indivdiual training example fed through the network wrt to each class.
 A_3 = sigmoid(Z_3);
 
# K, Number of classes.
 K = size(A_3, 2);

 for i=1:K
     # We want the ith column which corresponds to one class. Where each row in 
     # column i represents one training example fed forward through the network.
     A_3i = A_3(:,i);
     # One v all, for each iteration we need to construct Y_i which represents 
     # the training labels that correspond to the current class as 1 and all 
     # other classes as 0.
     y_i = (y == i);
     
     J += ((( -y_i'*log(A_3i)) - (1-y_i)'*log(1 - A_3i) )/(m) );
 endfor

# Calculating Regularization when lambda is not zero!
# Given, that this Neural Network will always have exactly 3 layers 
# Implies that we will have two theta matrices. However, the following
# code will generalize if more thetaX's are added to the list below.
if lambda != 0
    ThetaList = {Theta1, Theta2};

    SumThetas = 0;
    for j=1:size(ThetaList,2)
        # Remove bias parameter.
        Theta_j_noBias = ThetaList{j}(:,2:end);
        # Square all parameters and then sum.
        SumThetas += sum(sum(Theta_j_noBias.**2));     
    endfor
    
    regFactor = (lambda/(2*m))*SumThetas;
    J += regFactor;
endif
 
% ===================== MY CODE P1 -> END ========================
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% ===================== MY CODE P2 -> START ======================
D2 = 0;
D1 = 0;
for t=1:m
    ## *** Feed-Forward ***
    
    ## Select the t^th row from X now get its transpose this is now the t^th vector
    ## Note, bias units have already been added to X above.
    a_1 = X(t,:)';
    ## Note, the script that calls this code will leverage randInitializeWeights function
    ## and pass in Theta1 and Theta2 which we unpack and reform above.
    z_2 = Theta1 * a_1;
    # Adding bias units 
    a_2 = [1; sigmoid(z_2)];
    z_3 = Theta2 * a_2;
    a_3 = sigmoid(z_3);    

    ## *** Backpropagation ***
    
    ## Logical vector to identify y(t) as a member of one of the k class.
    z_2 = [1; z_2]; #bias
    y_k = (y(t) == [1:K]');
    
    ## When the cost = logistic regression then delta_L = a_L - y.
    delta_3 = (a_3 - y_k);

    ## This is one of the formulas of BP (see Apendix 2 - BP2).
    delta_2 = Theta2'*delta_3.*sigmoidGradient(z_2);
 
    ## Accumlate
    delta_2 = delta_2(2:end); ## Remove bias
 
    D2 += delta_3*a_2';
    D1 += delta_2*a_1';
endfor

## Implement Regularization here!
##Theta1(:,1) = 0;
##Theta2(:,1) = 0;
##Theta1_grad = D1./m + (lambda/m)*Theta1;
##Theta2_grad = D2./m + (lambda/m)*Theta2;

## Better way to implement Regularization.
##Theta1_grad = D1./m + [zeros(size(Theta1,1), 1), (lambda/m)*(Theta1(:, 2:end))];
##Theta2_grad = D2./m + [zeros(size(Theta2,1), 1), (lambda/m)*(Theta2(:, 2:end))];

Theta1_grad = D1./m;
Theta2_grad = D2./m;

Theta1_grad(:,2:end) += (lambda/m)*(Theta1(:, 2:end));
Theta2_grad(:,2:end) += (lambda/m)*(Theta2(:, 2:end)); 



% ===================== MY CODE P2 -> END ========================
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
