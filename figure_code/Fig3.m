clear; clc; close all
restoredefaultpath

count = 1;
res = 20;
mdsc = 'stress';

rep = 5;
ntime_samp = 50;

gamma_list = [-1,-0.5,-0.1,0,0.1,0.5,1];

colBar = 1;
for whichg = 1

parm.gammas  = 0.5; %gamma_list(whichg); %-2:.01:2;
parm.etas    = -0.001; %-.04:.001:.04;

init_w = 1e-10;

Y = [1 1 1 1 1 1 1 1
1 1 1 1 0 0 0 0
0 0 0 0 1 1 1 1
1 1 0 0 0 0 0 0
0 0 1 1 0 0 0 0
0 0 0 0 1 1 0 0
0 0 0 0 0 0 1 1
eye(4),zeros(4,4)
zeros(4,4),eye(4)];

X = eye(8,8); % input matrix

lr = 0.005; %%
parm.scale   = init_w;
parm.neuron  = 32;
parm.lrate   = lr;
parm.Nepochs = 3500;

E = Y*X';

parm.data = Y;
[U,S,V] = svd(E,'econ');

item = 1;
feature = 5;
nc = size(V,2);
%a feature m for item i receives a contribution from each mode u_m vi 
coeffs = U(feature,1:nc).*V(item,:);

Ni = size(Y,2); % obj (item), N1
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
    allmode = [];

    pair = 1;
    W1 = scale*randn(Nh,Ni)/sqrt(Nh);
    W2 = scale*randn(Nf,Nh)/sqrt(Nf);
    gamma = gvec(pair);
    eta = evec(pair);

    mode_components = [];
    err = nan(1,Nepochs);
    W1_norm = nan(1,Nepochs);
    W2_norm = nan(1,Nepochs);

    for ep = 1:Nepochs

        delta_W1_chl = zeros(size(W1));
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
%             delta_W1_chl = delta_W1_chl + 1/gamma*(hc-hf)*x';
            delta_W1_chl = delta_W1_chl + W2'*(y-y_hat)*x';
            delta_W2_chl = delta_W2_chl + y*hc' - y_hat*hf';

            if eta >= 0 
                dW1vec = bsxfun(@times, h_feedforward, bsxfun(@minus, x', diag(h_feedforward)*W1));
            else
                dW1vec = h_feedforward*x';
            end

            delta_W1_hebb = delta_W1_hebb + dW1vec;
        end % end of loop over objects

        mode_components(:,ep) = diag(U'*W2*W1*V);
        err(ep) = norm(Y-W2*W1,'fro').^2;
        W1_norm(ep) = norm(W1,'fro');
        W2_norm(ep) = norm(W2,'fro');
        rep_time(:,:,ep) = W1*X; % for MDS
        
        h_items = W1*X;
        all_i = 1;
        tmp = [];
        for i = 1:length(all_i)
            thisi = all_i(i);
            for j = 1:8
                tmp(j,i) = h_items(:,j)'*h_items(:,thisi)/norm(h_items(:,thisi))^2;
            end
        end
        gen(:,ep) = sum(tmp,2);
        
        %%%%%%%%%%%%%%%%
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
        %%%%%%%%%%%%%%%%
        
    
        h_features = W2';
        gen_f = [];
        for m = 14
            h_f = h_features(:,m);
            % h_f is a col vector
            gen_feats = h_f'*h_features/(h_f'*h_f);
            gen_f = [gen_f,gen_feats'];
        end
        tmp = sum(gen_f,2);
        proj_f(:,ep) = tmp;

        
    end % end of epoch
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    mode_components(:,ep+1) = diag(U'*W2*W1*V);
    err(ep+1) = norm(Y-W2*W1,'fro').^2;
    behav = W2*W1;
    Bcov = behav'*behav;
    Bcov = Bcov./max(Bcov(:));

    W1_norm(ep+1) = norm(W1,'fro');
    W2_norm(ep+1) = norm(W2,'fro');
    
    outW1(:,:,pair) = W1'*W1;
    allout(:,:,pair) = [err(samp); W1_norm(samp); W2_norm(samp)];
    allmode(:,:,pair) = mode_components(:,samp);

    disp('done')
    
    
    h_items = W1;
    for i = 1:8 % this item has new feature m = 1
    % to ask item j:
        for j = 1:8
            gen_all(i,j) = h_items(:,j)'*h_items(:,i)/norm(h_items(:,i))^2;
        end
    end
    
    h_features = W2';
    Nf = size(h_features,2);
    gen_f = [];
    for m = 1:Nf
        h_f = h_features(:,m);
        % h_f is a col vector
        gen_feats = h_f'*h_features/(h_f'*h_f);
        gen_f = [gen_f,gen_feats'];
    end
    
    ntol = size(rep_time,3);
    id = linspace(1,ntol,ntime_samp);
    keep_rep = rep_time(:,:,round(id));
    [nv,nob,nt] = size(keep_rep);
    t = 1:nt;
    points = reshape(keep_rep,[nv,nob*nt]);
    D = pdist(points','euclidean');
    D = D + rand(size(D))*eps;
    opts = statset('Display','iter');
    [MDS_y,eigvals] = mdscale(D,2,'Start','random','Replicates',rep,'Options',opts,'Criterion',mdsc);

    intRep = single(outW1);
    timecourse = single(allout);
    SV = single(allmode);
    ntime = size(SV,2);
    timevec = [1:res:parm.Nepochs-1,parm.Nepochs];
    SV = SV([1,2,4],:); % top,mid,bottom

    A = intRep; %reshape(intRep,[8,8,1,1]); % internal rep
    B = SV; % singular values
    C = reshape(timecourse,[3,ntime,1,1]); % error, w1, w2 timecourses

    fsize = 15;

    nrow = 1;
    np = 4;

    x = 1;
    y = 1;
    

    figure('position',[749   298   236   657])
    subplot(3,1,2)
    M = A(:,:,y,x);
    M = M./max(M(:));
    colormap(redblue)
    imagesc(M,[-1,1]);
    if colBar
    colorbar;
    end
    axis square
    
    set(gca,'xtick',1:8,'xticklabel',1:8)
    set(gca,'ytick',1:8,'yticklabel',1:8)
    if whichg==1
    xlabel('Objects'); ylabel('Objects'); 
    end
    set(gca,'fontsize',fsize)

    subplot(3,1,3)
    G_time = [];
    NVoxs = parm.neuron;
    G_end = [];
for i = 1:nob
    x_un = MDS_y(i:nob:end,1)';
    y_un = MDS_y(i:nob:end,2)';
   
    % rotate a bit
    theta = 0;
    x = [cos(theta) -sin(theta)]*[x_un; y_un];
    y = [sin(theta) cos(theta)]*[x_un; y_un];
    z = zeros(size(x));
%     z = z_un;
    surface([x;x],[y;y],[z;z],[t;t],...
            'facecol','interp',...
            'edgecol','interp',...
            'linew',2);
        hold on
        axis square

        G_end(i,:) = [x_un(:,end),y_un(:,end)];
        G_start(i,:) = [x_un(:,1),y_un(:,1)];
        colormap(gca,jet)
        
        if colBar
        colorbar('yticklabel',{[num2str(0)],...
            ['epoch:\newline',num2str(id(end))]},'ytick',[1,length(id)]);
        end
end
    hold on;
    G = G_end;
    lab = {'1','2','3','4','5','6','7','8'};
    text(G(:,1),G(:,2),lab,'VerticalAlignment','bottom','fontsize',fsize,'color','k')
    range = [min(G(:)),max(G(:))]*1.1;
    plot(G_start(:,1),G_start(:,2),'ks','markersize',5,'markerfacecolor','k')
    hold on;
    plot(G(:,1),G(:,2),'rs','markersize',5,'markerfacecolor','r')
    if whichg==1
    xlabel('dimension 1')
    ylabel('dimension 2')
    end
    set(gca,'fontsize',fsize)
    
    
    
    % error
    subplot(3,1,1)
    x = 1:id(end);
    y = err(1:end-1);
    z = zeros(size(x));
    t = x;
    surface([x;x],[y;y],[z;z],[t;t],...
            'facecol','interp',...
            'edgecol','interp',...
            'linew',2);
        if whichg==1
    xlabel('Time (Epochs)')
        end
    set(gca,'xtick','')
    axis square
    
    if whichg==1
    ylabel('sum squared error')
    end
    ylim([0,35])
    colormap(gca,jet)
    set(gca,'fontsize',fsize)
    title(['\gamma = ',num2str(parm.gammas(1)),', \eta = ',num2str(parm.etas(1))])
    
    if colBar
            colorbar('yticklabel',{[num2str(0)],...
            ['epoch:\newline',num2str(id(end))]},'ytick',[1,length(id)]);
    end
end

