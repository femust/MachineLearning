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

X = [ones(m, 1) X];


whole_sum=0;


a_1=sigmoid(X*Theta1');
m_a1 = size(a_1, 1);
a_1 = [ones(m_a1, 1) a_1];
h_x=sigmoid(a_1*Theta2');
Y = eye(num_labels);




for i=1:m
    a=-Y(y(i),:).*log(h_x(i,:))-(1-Y(y(i),:)).*log(1-h_x(i,:));
    whole_sum=whole_sum+sum(a);
end
J=(1/m)*whole_sum;


sum_theta1=sum(sum(Theta1(:,2:end).^2));
sum_theta2=sum(sum(Theta2(:,2:end).^2));

J=J+(lambda/(2*m))*(sum_theta1+sum_theta2);







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

delta_3=[];
delta_2=[];
z_2=[];
a_1=[];
a_2=[];
pre_a_2=[];
z_3=[];
a_3=[];
pom=[];
delta_accumulate_2=0
delta_accumulate_1=0;
for t=1:m
    t
    a_1(t,:)=X(t,:);
    z_2(t,:)=X(t,:)*Theta1';
    pre_a_2(t,:)=sigmoid(z_2(t,:));
    a_2(t,:)=[1 pre_a_2(t,:)];
    z_3(t,:)=a_2(t,:)*Theta2';
    a_3(t,:)=sigmoid(z_3(t,:));
    delta_3(t,:)=a_3(t,:)-Y(y(t),:);
    pom=delta_3(t,:)*Theta2;
    delta_2(t,:)=pom(2:end).*sigmoidGradient(z_2(t,:));
   delta_accumulate_2=delta_accumulate_2+delta_3(t,:)'*a_2(t,:); delta_3'*a_2 ;
   delta_accumulate_1=delta_accumulate_1+delta_2(t,:)'*X(t,:);
    
end
Theta1(:,1)=(zeros(1,size(Theta1,1))')
Theta2(:,1)=(zeros(1,size(Theta2,1))')
Theta1_grad=(1/m)*delta_accumulate_1+(lambda/m)*Theta1;
Theta2_grad=(1/m)*delta_accumulate_2+(lambda/m)*Theta2;
 




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
