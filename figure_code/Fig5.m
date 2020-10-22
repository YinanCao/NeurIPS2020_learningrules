clear;clc;close all
restoredefaultpath

figure('position',[556   509   405   307])

res = 20;
mdsc = 'stress';

ntime_samp = 25;

parm.gammas  = 1; 
parm.etas    = -.001;

init_w = 1e-10;
Y = [1 1 1 1 1 1 1 1
    1 1 1 1 0 0 0 0
    0 0 0 0 1 1 1 1
    1 1 0 0 0 0 0 0
    0 0 1 1 0 0 0 0
    0 0 0 0 1 1 0 0
    0 0 0 0 0 0 1 1
    eye(4),zeros(4,4)
    zeros(4,4),eye(4)]; % desired output matrix

X = eye(8,8); % input matrix
E = Y*X'; % input-ouput corr matrix

lr = 0.005; %%
parm.scale   = init_w;
parm.neuron  = 32;
parm.lrate   = lr;
parm.Nepochs = 10000;

parm.data = Y;
parm.freq = ones(1,size(parm.data,1));
Y = parm.data;
[U,S,V] = svd(E,'econ');

item = 1;
feature = 5;
nc = size(V,2);
%a feature m for item i receives a contribution from each mode u_m vi 
coeffs = U(feature,1:nc).*V(item,:);

Ni = size(X,1); % obj (item)
Nf = size(Y,1); % feature, N3
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
samp = [1:res:Nepochs-1,Nepochs]';
nsamp = length(samp);
gvec = ge_pair(:,1);
evec = ge_pair(:,2);

outW1 = nan(Ni,Ni,npair);
allout = nan(3,nsamp,npair);
allmode = [];%nan(Nf,nsamp,npair);

pair = 1;
W1 = scale*randn(Nh,Ni)/sqrt(Nh);
W2 = scale*randn(Nf,Nh)/sqrt(Nf);
gamma = gvec(pair);
eta = evec(pair);

mode_components = [];%nan(Nf,Nepochs);
err = nan(1,Nepochs);
W1_norm = nan(1,Nepochs);
W2_norm = nan(1,Nepochs);

count = 1;
cum_feat = [];
for ep = 1:Nepochs

    delta_W1_chl = zeros(size(W1)); % 100,8
    delta_W2_chl = zeros(size(W2));
    delta_W1_hebb = zeros(size(W1));

    for i = 1:Ni

        x = X(:,i);
        y = Y(:,i);

        % Calculate intermediate variables
        y_hat = W2*W1*x;
        h_feedforward = W1*x;
        hc = W1*x + gamma*W2'*y;
        hf = W1*x + gamma*(W2'*W2*W1*x);

        % Calculate weight updates
        delta_W1_chl = delta_W1_chl + W2'*(y-y_hat)*x';
        delta_W2_chl = delta_W2_chl + y*hc' - y_hat*hf';

        if eta >= 0
            dW1vec = bsxfun(@times, h_feedforward, bsxfun(@minus, x', diag(h_feedforward)*W1));
        else % no oja's norm
            dW1vec = h_feedforward*x';
        end
        delta_W1_hebb = delta_W1_hebb + dW1vec;
    end % end of loop over objects

    mode_components(:,ep) = diag(U'*W2*W1*V);
    err(ep) = norm(Y-W2*W1*X,'fro').^2;
    W1_norm(ep) = norm(W1,'fro');
    W2_norm(ep) = norm(W2,'fro');
    rep_time(:,:,ep) = W1;

    h_items = W1;
    all_i = 1; % left-most item
    tmp = [];
    for i = 1:length(all_i)
        thisi = all_i(i);
        for j = 1:8
            tmp(j,i) = h_items(:,j)'*h_items(:,thisi)/norm(h_items(:,thisi))^2;
        end
    end
    gen(:,ep) = tmp;


    % update Weights:
    if eta >= 0
        W1 = W1 + lr*delta_W1_chl + eta*delta_W1_hebb;
    else
        nn = [];
        for n = 1:size(W1,1)
            nn = [nn; norm(W1(n,:))^2];
        end
        W1 = W1 + bsxfun(@times,eta*(delta_W1_hebb),1./(nn+1)) + lr*delta_W1_chl;
    end

    W2 = W2 + lr * delta_W2_chl;

    h_features = W2';
    gen_f = [];
    for m = 14
        h_f = h_features(:,m); % the ?backpropagated? hidden representation of feature m
        % h_f is a col vector
        gen_feats = h_f'*h_features/(h_f'*h_f);
        gen_f = [gen_f,gen_feats'];
    end
    tmp = sum(gen_f,2);
    proj_f(:,ep) = tmp;
    
    cut_time = 5000;
    if ismember(ep,cut_time)
        gen_all = []; % each row is a new feature
        h_items = W1;
        for i = [1:8] % this item has new feature m = 1
        % to ask item j:
            for j = 1:8
                gen_all(i,j) = h_items(:,j)'*h_items(:,i)/norm(h_items(:,i))^2;
            end
        end
        y_orig = W2*W1*X;
        cum_feat = [cum_feat;y_orig;repmat(gen_all,10,1)];
        count = count+1;
    end

end % end of epoch
    
h_features = W2';
Nf = size(h_features,2);
gen_f = [];
for m = 1:Nf
    h_f = h_features(:,m); % the ?backpropagated? hidden represen- tation of feature m
    % h_f is a col vector
    gen_feats = h_f'*h_features/(h_f'*h_f);
    gen_f = [gen_f,gen_feats'];
end
    
ntol = size(rep_time,3);
id = linspace(1,ntol,ntime_samp);

keep_rep = rep_time(:,:,round(id));
[nv,nob,nt] = size(keep_rep);
t = 1:nt;

close all;
figure;
imagesc(gen_f,[-1,1]); colormap(redblue);colorbar

figure('position',[574   203   528   251])
subplot(121)
gen_to_plot = gen(1:8,1:end)';

imagesc([1 8],[0 max(t)],gen_to_plot,[-1,1])
hold on;
plot(1,cut_time(1), 'k<', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
plot(1,cut_time(end), 'k<', 'MarkerSize', 10, 'LineWidth', 2);
colormap(redblue)
colorbar
xlabel('Objects')
ylabel('Time (Epochs)')
set(gca,'FontSize',14,'xtick',1:8)

% close all;
subplot(122)
new = [cum_feat];
s = new'*new;
s = s./max(s(:));
imagesc(s,[-1,1]);axis square
colormap(redblue);
colorbar
xlabel('Objects')
ylabel('Objects')
set(gca,'FontSize',14,'ytick',1:8,'xtick',1:8)



