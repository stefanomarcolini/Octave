##
# author: stefano marcolini
# year:   2019
#
# A simple (basic) set of ml functions
# written in Octave that can be used for
# prediction and classification problems
#
# GENERIC:
#  - featureScaling
#
# PREDICTION:
#  - hXUnvectorized
#  - hXVectorized
#  - cost
#  - costFunction
#  - gradientDescent
#  - normalEquation
#
# CLASSIFICATION:
#  - sigmoid
#  - logisticCost
#  - logisticRegression
##

clc;


##
# FEATURE SCALING:
# REQUIRES: X(m,n) or V(n)
# MODIFIES:  ---
# EFFECT:   returns the scaled features
#           ie. 0 <= X(i,j) <= 1
##
function S = featureScaling(X)
clear functions;
  
  m = length(X(:,1));
  n = length(X(1,:));
  for i=1:n,
    if (r = (max(X(:,i)) - min(X(:,i)))) != 0,
      S(:,i) = (1 / r) * X(:,i);
    else
      S(:,i) = X(:,i);
    endif;
  endfor
  
clear functions;
endfunction;


##
# HX-UNVECTORIZED
# REQUIRES: X(m * n) containing the training examples
#           theta(n) theta values
# MODIFIES:  ---
# EFFECT:   returns the hypothesis of X
##
function H = hXUnvectorized(X, theta)
clear functions;
  % unvectorized implementation
  n = length(theta);
  m = length(X(:,:));
  H = zeros(m,1);
  
  for i=1:m,
    for j = 1:n,
      H(i) += theta(j) * X(i,j);
    end;
  end;
  
clear functions;
endfunction;


##
# HX-VECTORIZED
# REQUIRES: X(m * n) containing the training examples
#           theta(n) theta values
# MODIFIES:  ---
# EFFECT:   returns the hypothesis of X
##
function H = hXVectorized(X, theta)
clear functions;

  % vectorized implementation
  if length(X(:,1)) > length(X(1,:)),  % m >= n
    if iscolumn(theta),                 % theta is a column vector
      H = X*theta;
    else                                % theta is a row vector
      H = X * transpose(theta);
    end;
  else                                  % m < n
    if iscolumn(theta),                 % theta is a column vector
      H = transpose(X)*theta;
    else                                % theta is a row vector
      H = transpose(theta*X);
    end;
  end;  
 
clear functions;
endfunction;


##
# COST
# REQUIRES: X(m * n) containing the training examples
#           y(n)     dataset values
#           t(n)     theta values
# MODIFIES:  ---
# EFFECT:   returns the cost function J(theta)
#           J=1/(2*length(X(:,1))*sum((t'*X-y).^2);
##
function J = cost(X,y,t)
clear functions;

  m = length(X(:,1));            % number of trining examples
  n = length(X(1,:));            % number of features
  sqrErr = zeros(m,1);
  for i=1:m,
    htx = 0.0;
    for j=1:n,
      htx += t(j)*X(i,j);
    endfor;
    sqrErr(i) = (htx - y(i))^2;  % squared errors
  endfor;
             
  J = 1/(2*m)*sum(sqrErr);       % cost function values
  
clear functions;
endfunction;


##
# COST FUNCTION
# REQUIRES: X(m * n) containing the training examples
#           y(n)     dataset values
#           t(n)     theta values
#           vect     true -> use vectorized computation of hx
#                    false -> use unvectorized computation of hx
# MODIFIES:  ---
# EFFECT:   returns the cost function J(theta)
#           J=1/(2*size(X,1))*sum((transpose(t)*X-y).^2);
##
function J = costFunction(X,y,t,vect)
  
clear functions;

  m = size(X,1);               % number of trining examples
  
  if vect,
    P = hXVectorized(X,t);
  else
    P = hXUnvectorized(X,t);   % predictions of hypothesis on all m
  endif;
  
  sqrErr=(P-y).^2;             % squared errors
  J = 1/(2*m)*sum(sqrErr);     % cost function values
  
  
clear functions;
endfunction;


##
# GRADIENT DESCENT
# REQUIRES: X(m,n)   containing the training examples
#           y(m)     data result values (vertical vector)
#           t(n)     theta (vertical vector)
#           alpha    learning factor (es. 0.03)
#           feedback iterations after which to display feedback
#                    ie. iteration, theta, cost, minimum
# MODIFIES:  ---
# EFFECT:   returns the minimized cost function using
#           batched gradient descent
#           ie. the optimal theta values analitically
##
function G = gradientDescent(x,y,t,alpha,feedback)
clear functions

# =========================
# INNER HYPOTHESIS FUNCTION
# =========================
  # Hypothesis of x (vectorized implementation)
  function H = hXVect(X, T)
    if length(X(:,1)) > length(X(1,:)),      # m >= n
      if iscolumn(T),                        # theta is a column vector
        H = X*T;
      else                                   # theta is a row vector
        H = X * T';
      end;
    else                                     # m < n
      if iscolumn(T),                        # theta is a column vector
        H = X'*T;
      else                                   # theta is a row vector
        H = T'*X;
      end;
    end;
  endfunction;
  
# ===================
# INNER COST FUNCTION
# ===================
  # Cost of hypothesis of x (vectorized implementation)
  function J = costF(X,Y,T)
    m = size(X,1);                           # number of trining examples
    
    P = hXVect(X,T);                         # predictions of hypothesis on all m
    
    sqrErr=(P-y).^2;                         # squared errors
    J = 1/(2*m)*sum(sqrErr);                 # cost function values
  endfunction;
  
  
# ==============================
# GRADIENT DESCENT IMPLEMETATION
# ==============================
  m = length(y);
  n = length(x(1,:));
  c=0;
  tmp = t;
  while true,
    
    J = costF(x,y,t);                          # cost of x
    
    derivative = zeros(n,1);
    hx = x*t;
    for i=1:m,
      for j=1:n,
        derivative(j) += (hx(i)-y(i))*x(i,j);  # derivative part. 1
      endfor;
    endfor;
    
    derivative = derivative./ m;               # derivative end
    t = t - (alpha.*derivative);               # update theta simoultaneously
    
    Jm = costF(x,y,t);                         # new cost of x
    
    if c == 0 || mod(c, feedback) == 0,        # display feedback
      disp(c);
      disp(sprintf('t   : %0.20f\t', t'));
      disp(sprintf('cost: %0.20f\t', J));
      disp(sprintf('min : %0.20f\t', Jm));
    endif;
    
    if Jm < j,
      tmp = t;
    endif;
    
    if Jm >= J,                                # Quasi-Minimum conditions
      t = tmp;
      disp(c);
      disp("\nQUASI MINIMUM FOUND!");
      disp(sprintf('t   : %0.20f\t', t'));
      disp(sprintf('cost: %0.20f\t', Jm));
      disp(sprintf('min : %0.20f\t', J));
      if Jm > J,
        disp(sprintf('\nREASON: DIVERGING -> min != 0 && min > cost'));
      else
        disp(sprintf('\nREASON: STUCK -> min != 0 && min == cost'));
      endif;
      break;
    elseif Jm == 0,                            # Perfect-Minimum conditions
      disp(c);
      disp("\nMINIMUM FOUND!");
      disp(sprintf('t   : %0.20f\t', t'));
      disp(sprintf('cost: %0.20f\t', J));
      disp(sprintf('min : %0.20f\t', Jm));
      break;
    endif;
    
    ++c;                                       # increase counter for feedback
    
  endwhile;
  
   G = t;                                      # return theta for min cost

clear functions
endfunction;


##
# NORMAL EQUATION
# REQUIRES: X(m,n)   containing the training examples
#           y(m)     data result values
# MODIFIES:  ---
# EFFECT:   returns the normal equation
#           ie. the optimal theta values analitically
##
function G = normalEquation(X,y)
clear functions;

  #G = pinv(transpose(X)*X) * transpose(X) * y
  
  # X' = transpose(X)
  # \  = pinv(...)
  G = (X'*X) \ X' * y;
  
clear functions;
endfunction;