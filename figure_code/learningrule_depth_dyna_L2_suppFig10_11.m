clear; clc; close all;
% this code generates results for supp Fig. 10 and 11

% [R,~] = qr(randn(8,8));
% X = R; % input matrix, orthonormal
X = eye(8); % input matrix = one-hot column vectors

% eta sampling:
eta = 0.001;
gamma = .05;

nt = 100;
dt = .05;

lr = 0.01; % learning rate
neuron  = 32; % # neurons of hidden layer
Nepochs = 20000;
scale = 1e-10;

Y = [1 1 1 1 1 1 1 1
     1 1 1 1 0 0 0 0
     0 0 0 0 1 1 1 1
     1 1 0 0 0 0 0 0
     0 0 1 1 0 0 0 0
     0 0 0 0 1 1 0 0
     0 0 0 0 0 0 1 1
     eye(4),zeros(4,4)
     zeros(4,4),eye(4)];
Eyx = Y*X'; % input-output mapping 
[U,S,V] = svd(Eyx,'econ'); % SVD
             
F = ones(1,size(Y,1));
F = F/sum(F);

Ni = size(Y,2); % obj (item)
Nf = size(Y,1); % feature, N3
Nh = neuron; % number of neurons 

disp('computing start >>>>>>>>>')
disp(['learning rate = ',num2str(lr)])

W1 = scale*randn(Nh,Ni);
W2 = scale*randn(Nf,Nh);

% training epochs
D = 2;
count = 1;
for ep = 1:Nepochs

disp(num2str(ep))
delta_W1_chl = zeros(size(W1));
delta_W2_chl = zeros(size(W2));
delta_W1_hebb = zeros(size(W1));

% loop over objects:
for i = 1:Ni
    % Select input
    x = X(:,i);
    % target output
    y = Y(:,i);

    % free phase:
    x2 = zeros(Nf,nt);
    x1 = zeros(Nh,nt);
    for tt = 2:nt
        x1(:,tt) = x1(:,tt-1) + dt*(-x1(:,tt-1) + W1*x + gamma*W2'*x2(:,tt-1));
        x2(:,tt) = x2(:,tt-1) + dt*(-x2(:,tt-1) + W2*x1(:,tt-1));
    end
    x1_f = x1(:,end);
    x2_f = x2(:,end);
    free_h1(:,i) = x1_f;
    free_h2(:,i) = x2_f;


    % clamped phase:
    x1 = zeros(Nh,nt);
    for tt = 2:nt
        x1(:,tt) = x1(:,tt-1) + dt*(-x1(:,tt-1) + W1*x + gamma*W2'*y);
    end
    x1_c = x1(:,end);
    x2_c = y;
    clamp_h1(:,i) = x1_c;
    

    % weight update: CHL
    delta_W1_chl = delta_W1_chl + gamma^(1-D)*(x1_c*x'-x1_f*x');
    delta_W2_chl = delta_W2_chl + gamma^(2-D)*(x2_c*x1_c'-x2_f*x1_f');
    
    % Oja's rule when eta >= 0
    h_feedforward = W1*x;
    if eta >= 0
        dW1vec = bsxfun(@times, h_feedforward, bsxfun(@minus, x', diag(h_feedforward)*W1));
    else
        dW1vec = h_feedforward*x';
    end
    delta_W1_hebb = delta_W1_hebb + dW1vec;

end

Y_hat = W2*W1*X;
err(ep) = norm(Y-Y_hat,'fro').^2; % frobenius norm^2
mode_components(:,ep) = diag(U'*W2*W1*V); % mode strength
    
if ismember(ep, [1,100,500,1000,2000,3000,4000,5000,10000,15000,20000])
    h = [];
    h{1} = W1*X;
    h{2} = W2*W1*X;
    tmp = [];
    for l = 1:length(h)
        tmp(:,:,l) = h{l}'*h{l};
    end
    internal_rep(:,:,:,count) = tmp;
    F_rep(:,:,1,count) = free_h1'*free_h1;
    F_rep(:,:,2,count) = free_h2'*free_h2;
    C_rep(:,:,1,count) = clamp_h1'*clamp_h1;
    count = count + 1;
end

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

if isnan(W1(1,1))
    break;
end
% 
% hold on;
% plot(ep,err(ep),'r.')
% hold on;
% ylim([0,35])
% xlim([0,Nepochs])
% pause(0.001)

end % end of epoch

disp('done')

close all
hold on;
plot(err,'linewidth',2)
hold on;
ylabel('error')
% plot(d,'linewidth',2)
ylabel('Training error');
xlabel('Epoch')
set(gca,'fontsize',20,'box','off','linewidth',2)
ylim([0,35])


c = clock;
c = ['_',num2str(c(1)),'_',num2str(c(2)),'_',num2str(c(3)),'_',num2str(c(4)),'_',num2str(c(5))];
save(['/Users/yinancaojake/Desktop/NeurIPS/code/depth/D2_neuron',num2str(neuron),c,'.mat'],'err','gamma','eta','lr','nt','dt','scale',...
    'mode_components','internal_rep','F_rep','C_rep')


%%


clear; clc; close all;

fn = {'D2.mat','D3.mat'};
close all

figure('position',[360   675   801   280])
for depth = 2
    load(fn{depth})
    
    subplot(1,2,1)
    plot(err,'linewidth',2)
    hold on;
    ylabel('Training error');
    xlabel('Epoch')
    set(gca,'fontsize',20,'box','off','linewidth',2)
    
    subplot(1,2,2)
    plot(mode_components','linewidth',2)
    hold on;
    ylabel('Mode strength');
    xlabel('Epoch')
    set(gca,'fontsize',20,'box','off','linewidth',2)
    
end

time_str = [1,100,500,1000,2000,3000,4000,5000,10000];
size(internal_rep)
figure('position',[360   675   801   280])
nlayer = size(internal_rep,3);
ntime = size(internal_rep,4);
m = permute(internal_rep,[1,2,4,3]);
m = reshape(m,[8,8,nlayer*ntime]);
for i = 1:size(m,3)
    subplot(nlayer,ntime,i)
    out = m(:,:,i);
%     out = out./max(abs(out(:)));
    imagesc(out); axis square
    if i <= length(time_str)
    title(['t = ',num2str(time_str(i))])
    end
%     colormap(hot)
%     colorbar
end

% close all
% hold on;
% plot(sqrt(err),'linewidth',2)
% hold on;
% ylabel('error')
% plot(d,'linewidth',2)
% ylabel('Training error');
% xlabel('Epoch')
% set(gca,'fontsize',20,'box','off','linewidth',2)
% % xlim([0,800])
% 
% 
% 
