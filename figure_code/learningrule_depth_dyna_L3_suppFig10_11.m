clear; clc; 
close all;
% this code generates results for supp Fig. 11

% [R,~] = qr(randn(8,8));
% X = R; % input matrix, orthonormal
X = eye(8); % input matrix = one-hot column vectors

% eta sampling:
eta = 0.001;
gamma = .05;

nt = 100;
dt = .5;

lr = 0.01; % learning rate
neuron  = 32; % # neurons of hidden layer
Nepochs = 20000;
scale = 1e-3;

Y = [1 1 1 1 1 1 1 1 % target output matrix, each col = one object
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
W3 = scale*randn(Nf,Nh);
W2 = scale*randn(Nh,Nh);

% training epochs
count = 1;
D = 3;
for ep = 1:Nepochs
    disp(num2str(ep))
    delta_W1_chl = zeros(size(W1));
    delta_W2_chl = zeros(size(W2));
    delta_W3_chl = zeros(size(W3));
    delta_W1_hebb = zeros(size(W1));
    delta_W2_hebb = zeros(size(W2));

    % loop over objects:
    for i = 1:Ni
        % Select input
        x = X(:,i);
        % target output
        y = Y(:,i);

        % free phase:
        x3 = zeros(Nf,nt);
        x2 = zeros(Nh,nt);
        x1 = zeros(Nh,nt);
        for tt = 2:nt
            x1(:,tt) = x1(:,tt-1) + dt*(-x1(:,tt-1) + W1*x + gamma*W2'*x2(:,tt-1));
            x2(:,tt) = x2(:,tt-1) + dt*(-x2(:,tt-1) + W2*x1(:,tt-1) + gamma*W3'*x3(:,tt-1));
            x3(:,tt) = x3(:,tt-1) + dt*(-x3(:,tt-1) + W3*x2(:,tt-1));
        end
        x1_f = x1(:,end);
        x2_f = x2(:,end);
        x3_f = x3(:,end);
        
        free_h1(:,i) = x1_f;
        free_h2(:,i) = x2_f;
        free_h3(:,i) = x3_f;

        % clamped phase:
        x1 = zeros(Nh,nt);
        x2 = zeros(Nh,nt);
        x3_c = y;
        for tt = 2:nt
            x2(:,tt) = x2(:,tt-1) + dt*(-x2(:,tt-1) + W2*x1(:,tt-1) + gamma*W3'*x3_c);
            x1(:,tt) = x1(:,tt-1) + dt*(-x1(:,tt-1) + W1*x + gamma*W2'*x2(:,tt-1));
        end
        x1_c = x1(:,end);
        x2_c = x2(:,end);
        clamp_h1(:,i) = x1_c;
        clamp_h2(:,i) = x2_c;
        

        % weight update: chl
        delta_W1_chl = delta_W1_chl + gamma^(1-D)*(x1_c*x'-x1_f*x');
        delta_W2_chl = delta_W2_chl + gamma^(2-D)*(x2_c*x1_c'-x2_f*x1_f');
        delta_W3_chl = delta_W3_chl + gamma^(3-D)*(x3_c*x2_c'-x3_f*x2_f');
        
        % Oja's rule when eta >= 0
        if eta >= 0
            dW1vec = bsxfun(@times, W1*x, bsxfun(@minus, x', diag(W1*x)*W1));
            dW2vec = bsxfun(@times, W2*W1*x, bsxfun(@minus, (W1*x)', diag(W2*W1*x)*W2));
        else
            dW1vec = (W1*x)*x';
            dW2vec = (W2*W1*x)*(W1*x)';
        end
        delta_W1_hebb = delta_W1_hebb + dW1vec;
        delta_W2_hebb = delta_W2_hebb + dW2vec;

    end

    Y_hat = W3*W2*W1*X;
    err(ep) = norm(Y-Y_hat,'fro').^2; % frobenius norm^2
    mode_components(:,ep) = diag(U'*W3*W2*W1*V); % mode strength
    
if ismember(ep, [1,100,500,1000,2000,3000,4000,5000,10000,15000,20000])
    h = [];
    h{1} = W1*X;
    h{2} = W2*W1*X;
    h{3} = W3*W2*W1*X;
    tmp = [];
    for l = 1:length(h)
        tmp(:,:,l) = h{l}'*h{l};
    end
    internal_rep(:,:,:,count) = tmp;
    
    F_rep(:,:,1,count) = free_h1'*free_h1;
    F_rep(:,:,2,count) = free_h2'*free_h2;
    F_rep(:,:,3,count) = free_h3'*free_h3;
    C_rep(:,:,1,count) = clamp_h1'*clamp_h1;
    C_rep(:,:,2,count) = clamp_h2'*clamp_h2;
    
    count = count + 1;
end
    

    if eta >= 0
        W1 = W1 + lr*delta_W1_chl + eta*delta_W1_hebb;
        W2 = W2 + lr*delta_W2_chl + eta*delta_W2_hebb;
    else
        nn = [];
        for n = 1:size(W1,1)
            nn = [nn; norm(W1(n,:))^2];
        end
        a = bsxfun(@times,eta*(delta_W1_hebb),1./(nn+1)) + lr*delta_W1_chl;
        W1 = W1 + a;
        
        nn = [];
        for n = 1:size(W2,1)
            nn = [nn; norm(W2(n,:))^2];
        end
        a = bsxfun(@times,eta*(delta_W2_hebb),1./(nn+1)) + lr*delta_W2_chl;
        W2 = W2 + a;
        
    end
    
    W3 = W3 + lr*delta_W3_chl;
    
    if isnan(W1(1,1))
        break;
    end

%     hold on;
%     plot(ep,err(ep),'b.')
%     hold on;
%     ylim([0,35])
%     xlim([0,Nepochs])
%     pause(0.0001)
%     
%     subplot(1,3,2)
%     hold on;
%     M = W1*X; 
%     imagesc(M'*M);
%     hold on;
%     pause(0.001)
    
%     subplot(1,3,3)
%     M = W2*W1*X; hold on;
%     imagesc(M); axis square
%     hold on;
%     pause(0.001)

end % end of epoch


disp('done')

c = clock;
c = ['_',num2str(c(1)),'_',num2str(c(2)),'_',num2str(c(3)),'_',num2str(c(4)),'_',num2str(c(5))];
save(['/Users/yinancaojake/Desktop/NeurIPS/code/depth/D3_neuron',num2str(neuron),c,'.mat'],'err','gamma','eta','lr','nt','dt','scale',...
    'mode_components','internal_rep','F_rep','C_rep')

%%

close all
figure;
subplot(1,2,1)
plot(err,'linewidth',2)
ylabel('Training error');
xlabel('Epoch')
set(gca,'fontsize',20,'box','off','linewidth',2)
ylim([0,35])

subplot(1,2,2)
plot(mode_components','linewidth',1)
ylabel('Strength of mode');
xlabel('Epoch')
set(gca,'fontsize',20,'box','off','linewidth',2)
ylim([0,4])

addpath(genpath('/Users/yinancaojake/Documents/MATLAB/YC_private/'))
clc;
M = F_rep;
% M = internal_rep;
size(M)
nl = size(M,3);
nt = size(M,4);
figure
for l = 1:nl
    for t = 1:size(M,4)
        m = M(:,:,l,t);
        m = m./max(abs(m(:)));
        subplot(nl,size(M,4),nt*(l-1) + t)
        imagesc(m,[-1,1]); axis square
        colormap(redblue);
        set(gca,'xtick',[],'ytick',[])
    end
end

% %%
% 
% save(['D3_lr=',num2str(lr),'.mat'],'err','gamma','eta','lr','nt','dt','scale',...
%     'mode_components','internal_rep')
% 
% % 
% % plot(err,'linewidth',2)
% % hold on;
% % ylabel('error')
% % 
% 
% close all
% hold on;
% plot((err),'linewidth',2)
% hold on;
% ylabel('error')
% % plot(d,'linewidth',2)
% ylabel('Training error');
% xlabel('Epoch')
% set(gca,'fontsize',20,'box','off','linewidth',2)
% % xlim([0,6000])