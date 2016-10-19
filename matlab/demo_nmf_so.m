function demo_nmf_so

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

% the number of components
K = 2;

% the number of optimization from different initializations
opts.reps   = 5;
% the maximum number of updates
opts.itrMax = 300;
% weight for orthogonality (0 <= wo <= 1)
opts.wo = 0.05;



% initialize the random number generater
s = RandStream('mt19937ar','Seed',0);
RandStream.setGlobalStream(s)

% NMF for X
[C, S] = nmf_so(X, K, opts);



%%% display results
figure
set(gcf,'Position',[100 100 1000 400])
for k=1:K
  subplot(1,3,k)
  X = reshape( C(:,k)',xdim, ydim);
  imagesc(X)
  colormap gray
  axis off
  axis equal
  title(strcat(['Component ', num2str(k)]))
  ylim([1,xdim])
  xlim([1,ydim])
end


subplot(1,3,3)
for k = 1:size(S,2)
    S(:,k) = S(:,k)/sqrt(sum(S(:,k).^2));
end
plot(S);
xlim([1,length(S)])
xlabel('Channel')
title('EELS spactra')

