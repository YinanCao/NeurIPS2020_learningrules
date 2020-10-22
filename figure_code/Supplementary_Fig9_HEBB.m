% weight change.

clear; clc; close all
addpath(genpath('/Users/yinancaojake/Documents/MATLAB/YC_private/'))

% eta sampling:
eta_all = [-0.1, -0.001, 0, 0.001, 0.1];
gamma_all = 0;

init_w = 1e-10;
wstr = {'small_normsq_ultimate';'large'};

for whichweight = 1:length(init_w)

lr = 0.005;
parm.scale   = init_w(whichweight);

parm.neuron  = 32;
parm.lrate   = lr;
parm.Nepochs = 10000;
parm.gammas  = gamma_all;
parm.etas    = eta_all;
E = [1 1 1 1 1 1 1 1
     1 1 1 1 0 0 0 0
     0 0 0 0 1 1 1 1
     1 1 0 0 0 0 0 0
     0 0 1 1 0 0 0 0
     0 0 0 0 1 1 0 0
     0 0 0 0 0 0 1 1
     eye(4),zeros(4,4)
     zeros(4,4),eye(4)];
parm.data = E;
             
Y = parm.data;
[U,S,V] = svd(Y,'econ');

Ni = size(Y,2); % obj (item), N1
Nf = size(Y,1); % feature, N3
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
samp = [1:Nepochs-1,Nepochs]';
nsamp = length(samp);

% outW1 = nan(Ni,Ni,npair);
allout = nan(3,nsamp,npair);
allmode = nan(Nrank,nsamp,npair);

gvec = ge_pair(:,1);
evec = ge_pair(:,2);

for pair = 1:size(ge_pair,1)
       if ~mod(pair,10)
           disp(['>>>>>> ',num2str(pair),' >>>>>>'])
       end
        W1 = scale*randn(Nh,Ni)/sqrt(Nh);
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

            for i = 1:Ni
                % Select input
                x = zeros(Ni,1);
                x(i) = 1;
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
                
                % Oja's rule when eta > 0
                if eta >= 0
                    dW1vec = bsxfun(@times, h_feedforward, bsxfun(@minus, x', diag(h_feedforward)*W1));
                else
                    dW1vec = h_feedforward*x';
                end
                delta_W1_hebb = delta_W1_hebb + dW1vec;

            end
            mode_components(:,ep) = diag(U'*W2*W1*V);
            err(ep) = norm(Y-W2*W1,'fro').^2;
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

ntime = size(allweight,2);
ng = length(parm.gammas);
ne = length(parm.etas);
C = reshape(allweight,[4,ntime,ne,ng]); % error, w1, w2 timecourses






end


%
close all;

figure('position',[124         573        1531         382]);
timevec = 1:parm.Nepochs;
np = size(C,3);
for cc = 1:np

    W_s = cumsum(C([1,2,3,4],:,cc)');
    subplot(1,np,cc)
    ccc = {'r','b','m','g'};
    linestyle = {'-.',':','-','--'};
    ww = [1,2,2,1];
    for k = 1:4

        plot(timevec,W_s(:,k),linestyle{k},'color',ccc{k},'linewidth',ww(k))
        hold on;
    end

    title(['\gamma = ',num2str(round(parm.gammas(1),1)),', \eta = ',num2str(round(parm.etas(cc),3))])
    axis square
    set(gca,'fontsize',16,'box','off')

    if cc ==1
        ylabel({'Synaptic strength'; 'change'})
        
    end
    
    if cc==1
        xlabel('Time (Epochs)')
    end
end
