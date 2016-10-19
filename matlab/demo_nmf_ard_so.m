function demo_nmf_ard_so

% load theoretical data of Mn3O4 without noise
load ./mn3o4_f2.mat
ximage = datar;
clear datar
scale_spect = max(ximage(:));

% focusing channel
n_ch = 37:116;
ximage = ximage(:,:,n_ch);

% # of pixels along x and y axis, # of EELS channels
[xdim,ydim,Nch] = size(ximage);

% generating pahtom data by adding gaussian noise
X = reshape(ximage, xdim*ydim, Nch);
s2_noise = 0.1;  %noise variance
X = X + randn(size(X))*s2_noise*scale_spect;
X = (X + abs(X))/2;
scale_X = mean(X(:));
X = X / scale_X;


% the maximum number of components
K = 10;

% # of pixels along x and y axis, # of EELS channels
[xdim,ydim,Nch] = size(ximage);

% the number of optimization from different initializations
opts.reps   = 5;
% the maximum number of updates
opts.itrMax = 5*10^3;
% weight for orthogonality (0 <= wo <= 1)
opts.wo = 0.05;
% sparse priors (1: L1(expornential pdf),  2: L2(half Gaussian pdf))
L = 1;
opts.sparse_type = L;
%hyper parameter of p(lambda|a,b)
opts.a           = 1/L + eps;

  
% initialize the random number generater
s = RandStream('mt19937ar','Seed',0);
RandStream.setGlobalStream(s)


% NMF for X
[C, S, lambda, obj, lambdas] = nmf_ard_so(X, K, opts);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% display results
kk = sum(C)>eps;
C = C(:,kk);
S = S(:,kk);
Kr = size(C,2);

for k=1:Kr
  figure
  X = reshape( C(:,k)',xdim, ydim);
  imagesc(X)
  colormap gray
  axis off
  axis equal
  ylim([1,xdim])
  xlim([1,ydim])
  title(strcat(['Component ', num2str(k)]))
end

figure
plot(S(:,1:Kr),'LineWidth',2);
set(gca,'FontName', 'Helvetica', 'FontSize',16)
xlabel('Channel')
ylabel('Intensity')
legend( strsplit( num2str((1:Kr)) ) )


figure
plot(lambdas,'LineWidth',2)
set(gca,'FontName', 'Helvetica', 'FontSize',16)
xlabel('Iterations')
xlim([1,size(lambdas,1)])
ylabel('\lambda')
legend( strsplit( num2str((1:size(lambdas,2))) ) )
