function [C_best, S_best, lambda_best, obj_best, lambdas_best,opts] = nmfard(X, K, opts)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% NMF (Nonnegative Matrix Factorization) 
%  with ARD (Automatic Relevance Determination) prior
%
%                       (c) Motoki Shiga, Gifu University, Japan
%
%-- INPUT --------------------------------------------------------------
%
%   X    : matrix with the size of (Nxy x Nch)
%          Nxy: the number of measurement points on specimen
%          Nch: the number of spectrum channels
%   K    : the number of components
%   opts : options
%
%-- OUTPUT -------------------------------------------------------------
%
%   C_best   : densities of components at each point
%   S_best   : spectrums of components
%   obj_best : learning curve (error value after each update)
%   
%
%  Reference
%  [1] Motoki Shiga, Kazuyoshi Tatsumi, Shunsuke Muto, Koji Tsuda, Yuta Yamamoto, Toshiyuki Mori, Takayoshi Tanji, 
%      "Sparse Modeling of EELS and EDX Spectral Imaging Data by Nonnegative Matrix Factorization",
%      Ultramicroscopy, Vol.170, p.43-59, 2016.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Options
reps        = opts.reps;        % the number of initializations
itrMax      = opts.itrMax;      % the maximum number of updates
sparse_type = opts.sparse_type; % sparse priors
                                %(1: L1(expornential pdf),  2: L2(half Gaussian pdf))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[Nxy, Nch] = size(X);

wo = opts.wo;

%set hyper-parameters
a    = opts.a;
mu_X = mean(X(:));
if sparse_type==1
  b = mu_X*(a-1)*sqrt(Nch)/K;
  c = (Nch+a+1);
elseif sparse_type==2
  tmp = exp( 2*(gammaln(a) - gammaln(a-0.5)) );
  b = pi*mu_X^2*Nch/2/K^2 * tmp;
  c = (0.5*Nch+a+1);
end
opts.b = b;
opts.c = c;
  
obj_best = inf;
mean_X = sum(X(:))/(Nxy*Nch); % Data sample mean per component
disp(' ')
disp('Running NMF with ARD and Soft Orthogonal constraint....')
disp('Optimizing components from different initializations:')
for rep = 1:reps
  disp( strcat([num2str(rep),' / ',num2str(reps)]) )
  % Initialization of matrix C
  C = (rand(Nxy,K) + 1)*(sqrt(mean_X/K));  
  % Initialization of matrix S
  i = randsample(Nxy,K);
  S = X(i,:)';
  for j = 1:K
    S(:,j) = S(:,j) / (S(:,j)'*S(:,j)); % normalization
  end
  % Initialization of lambda
  lambda = (sum(C) + b)/c;
  %update C and S by HALS algorithm
  [C,S,lambda,obj,lambdas]...
    = nmfard_update(X,C,S,lambda,wo,a,b,sparse_type,K,itrMax);
  %choose the best optimization result
  if obj(end) < obj_best(end)
    C_best   = C;    S_best   = S;  lambda_best = lambda;  
    obj_best = obj;  lambdas_best = lambdas;
  end
end
%remove small values
C_best(C_best<eps) = 0;   S_best(S_best<eps) = 0;
%sort component by the order of spectral peak positions 
[lambda_best,k] = sort(lambda_best,2,'descend');
C_best = C_best(:,k);   S_best = S_best(:,k);  
lambdas_best = lambdas_best(:,k);

disp('Finish the optimization of our model!')
disp(' ')
end


function [C, S, lambda, obj,lambdas]...
  = nmfard_update(X,C,S,lambda,wo,a,b,sparse_type,K,itrMax)

const = K*(gammaln(a) - a*log(b));
sigma2 = mean(mean(X.^2));

[Nxy, Nch] = size(X);
obj     = nan(itrMax,1);
lambdas = nan(itrMax,K);
cj = sum(C,2);
for itr = 1:itrMax
  
  %update S
  XC = X'*C;
  CC = C'*C;
  for j = 1:K
    S(:,j) = XC(:,j) - S*CC(:,j)+ CC(j,j)*S(:,j);
    S(:,j) = (S(:,j) + abs(S(:,j)))/2;
    c = sqrt(S(:,j)'*S(:,j));
    if c>0
      S(:,j) = S(:,j) / c;
    else
      S(:,j) = 1/sqrt(Nch);
    end
  end
  
  %update C
  XS = X*S;
  SS = S'*S;
  for j = 1:K
    cj     = cj - C(:,j);
    C(:,j) = XS(:,j) - C*SS(:,j)+ SS(j,j)*C(:,j);
    if sparse_type==1
      C(:,j) = C(:,j) - sigma2/lambda(j);
    elseif sparse_type==2
      C(:,j) = C(:,j) /(1 + sigma2*lambda(j)^(-1));
    end
    if wo>0
      C(:,j) = C(:,j) - wo*(cj'*C(:,j))/(cj'*cj)*cj;
    end
    C(:,j) = (C(:,j) + abs(C(:,j)))/2;
    cj     = cj + C(:,j);
  end
  
  if itr>3
    % merge if S are almost same
    [C,S]=merge_same_S(C,S);
  end
  
  if sum(cj)<eps
    C(:,:) = eps;
  end
    
  %update lambda(ARD parameters)
  if sparse_type==1
    lambda = (sum(C) + b)/(Nxy+a+1) + eps;
  elseif sparse_type==2
    lambda = (0.5*sum(C.^2) + b)/(0.5*Nxy+a+1) + eps;
  end
  
  %trajectory of lambda
  lambdas(itr,:) = lambda;
  
  %update sigma2
  X_est = C*S';  %reconstracted data matrix
  sigma2 = mean(mean((X-X_est).^2));
    
  %cost
%   X_est = C*S';  %reconstracted data matrix
  obj(itr) = Nxy*Nch/2*log(2*pi*sigma2) + Nxy*Nch/2;  %MSE
  if sparse_type==1
    obj(itr) = obj(itr) + (lambda.^(-1))*(sum(C)+b)'...
                        + (Nxy+a+1)*sum(log(lambda)) + const;
%                         + (Nch+a+1)*sum(log(lambda)) + const;
  elseif sparse_type==2
    obj(itr) = obj(itr) - K*Nxy/2*log(2/pi) + (0.5*Nxy+a+1)*sum(log(lambda))...
                + (lambda.^(-1))*(sum(C.^2)/2+b)' - const;
  end

  %cheack convergence
  if (itr>1) && (abs(obj(itr-1) - obj(itr)) < eps)
    obj     = obj(1:itr);
    lambdas = lambdas(1:itr,:);
    break;
  end

end

end


function [C,S]=merge_same_S(C,S)

theta = 0.99; %thereshold for merge

SS = S'*S;
[i,j] = find(SS>theta);
m = i<j;
i = i(m);  j = j(m);

Nch = size(S,1);
for n = 1:length(i)
  S(:,j(n)) = 1/sqrt(Nch);
  C(:,i(n)) = sum(C(:,[i(n),j(n)]),2);
  C(:,j(n)) = 0;  
end


end


