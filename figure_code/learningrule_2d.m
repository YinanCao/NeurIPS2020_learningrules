clear; clc; close all
% note: this code generates the data in Fig1 and Fig6

[R,~] = qr(randn(8,8));
X = R; % input matrix, orthonormal
% X = eye(8); % input matrix = one-hot column vectors

% target output matrix, each col = one object
% each row is a feature:
Y = [1 1 1 1 1 1 1 1 
     1 1 1 1 0 0 0 0
     0 0 0 0 1 1 1 1
     1 1 0 0 0 0 0 0
     0 0 1 1 0 0 0 0
     0 0 0 0 1 1 0 0
     0 0 0 0 0 0 1 1
     eye(4),zeros(4,4)
     zeros(4,4),eye(4)];

% eta/gamma sampling:
s = linspace(-5,log10(0.8),80);
s = 10.^s;
s = [-s,0,s];
s = sort(s);
eta_all = s;
gamma_all = sort(unique([-1.5:.04:1.5,0])); % add 0

myCluster = parcluster('local');
ndesiredworkers = myCluster.NumWorkers;
p = gcp('nocreate');
if ~isempty(p)
    Nwork = p.NumWorkers;
else
    Nwork = 0;
end
if Nwork ~= ndesiredworkers
  delete(gcp('nocreate'))
  parpool('local', ndesiredworkers, 'IdleTimeout', 30);
end

% small vs. large inititialization of weight matrices
init_w = [1e-10; .1];
wstr = {'small_normsq_orthX';'large_normsq_ultimate'};

for whichweight = 1:length(init_w)

lr = 0.005; % learning rate
parm.lrate   = lr;
parm.scale   = init_w(whichweight);
parm.neuron  = 32; % neurons of hidden layer
parm.Nepochs = 10000; % number of training epochs
parm.gammas  = gamma_all;
parm.etas    = eta_all;
parm.outputdata = Y;
Eyx = Y*X'; % input-output mapping 
[U,S,V] = svd(Eyx,'econ'); % SVD
            
[Nf, Ni] = size(Y); % [output neurons, items]
[Nx, ~]  = size(X); % input neurons
Nrank = min([Ni,Nf]);
Nh = parm.neuron; % number of neurons in the hidden layer, N2
scale = parm.scale;
Nepochs = parm.Nepochs;
allgamma = parm.gammas;
alleta = parm.etas;

disp('computing start >>>>>>>>>')
disp(['learning rate = ',num2str(lr)])

ge_pair = [];
for gi = 1:length(allgamma)
    for ei = 1:length(alleta)
        gamma = allgamma(gi); % Feedback scale
        eta = alleta(ei); % Hebbian term
        ge_pair = [ge_pair; gamma, eta];
    end
end

npair = size(ge_pair,1);
samp = [1:10:Nepochs-1,Nepochs]';
nsamp = length(samp);

allout = nan(3,nsamp,npair);
allmode = nan(Nrank,nsamp,npair);

gvec = ge_pair(:,1);
evec = ge_pair(:,2);

parfor pair = 1:size(ge_pair,1)
       if ~mod(pair,10)
           disp(['>>>>>> ',num2str(pair),' >>>>>>'])
       end
        W1 = scale*randn(Nh,Nx)/sqrt(Nh);
        W2 = scale*randn(Nf,Nh)/sqrt(Nf);
        gamma = gvec(pair);
        eta = evec(pair);

        mode_components = nan(Nrank,Nepochs);
        err = nan(1,Nepochs);
        W1_norm = nan(1,Nepochs);
        W2_norm = nan(1,Nepochs);
        delta_w = nan(4,Nepochs);

        for ep = 1:Nepochs

            delta_W1_chl = zeros(size(W1)); % 100,8
            delta_W2_chl = zeros(size(W2));
            delta_W1_hebb = zeros(size(W1));

            for i = 1:Ni % loop over objects
                % Select input
                x = X(:,i);
                % target output
                y = Y(:,i);

                % Calculate intermediate variables
                y_hat = W2*W1*x;
                h_feedforward = W1*x;
                hc = W1*x + gamma*W2'*y;
                hf = W1*x + gamma*(W2'*W2*W1*x);

                % Calculate weight updates
                % delta_W1_chl = delta_W1_chl + 1/gamma*(hc-hf)*x';
                delta_W1_chl = delta_W1_chl + W2'*(y-y_hat)*x';
                delta_W2_chl = delta_W2_chl + y*hc' - y_hat*hf';

                % Oja's rule when eta >= 0
                if eta >= 0
                    dW1vec = bsxfun(@times, h_feedforward, bsxfun(@minus, x', diag(h_feedforward)*W1));
                else
                    dW1vec = h_feedforward*x';
                end
                delta_W1_hebb = delta_W1_hebb + dW1vec;

            end
            mode_components(:,ep) = diag(U'*W2*W1*V); % mode strength
            err(ep) = norm(Y-W2*W1*X,'fro').^2; % frobenius norm^2
            W1_norm(ep) = norm(W1,'fro');
            W2_norm(ep) = norm(W2,'fro');
            
            st_w1_chl = norm(lr*delta_W1_chl,'fro');
            st_w2_chl = norm(lr*delta_W2_chl,'fro');
            

            if eta >= 0
                W1 = W1 + lr*delta_W1_chl + eta*delta_W1_hebb;
                st_w1_h  = norm(eta*delta_W1_hebb,'fro');
                st_w1_tol = norm(lr*delta_W1_chl + eta*delta_W1_hebb, 'fro');
            else
                nn = [];
                for n = 1:size(W1,1)
                    nn = [nn; norm(W1(n,:))^2];
                end
                a = bsxfun(@times,eta*(delta_W1_hebb),1./(nn+1)) + lr*delta_W1_chl;
                W1 = W1 + a;
                st_w1_h  = norm(bsxfun(@times,eta*(delta_W1_hebb),1./(nn+1)),'fro');
                st_w1_tol = norm(a,'fro');
            end
            W2 = W2 + lr*delta_W2_chl;
            
            delta_w(:,ep) = [st_w1_chl,st_w1_h,st_w1_tol,st_w2_chl];
            
        end % end of epoch

        allout(:,:,pair) = [err(samp); W1_norm(samp); W2_norm(samp)];
        allmode(:,:,pair) = mode_components(:,samp);
        allweight(:,:,pair) = delta_w(:,samp);
end
disp('done')

timecourse = single(allout);
SV = single(allmode);
allweight = single(allweight);

outdir = '/results/';
fn = datestr(clock);
fn(strfind(fn,' '))='-';
fn(strfind(fn,':'))='-';
fn = [wstr{whichweight},'_Oja_finer_fast_chl_hebb_gd_lr=',num2str(lr),'_',fn,'.mat'];
if ~exist(outdir,'dir')
    mkdir(outdir)
end
save([outdir,fn],'parm','timecourse','SV','allweight');
disp('saved >>>>>>>>>>>>>>>')

end

delete(gcp('nocreate'))

