clear; clc;close all;
restoredefaultpath
d1 = 'initialW=small.mat';
% download the saved statistics here: https://osf.io/encym/?view_only=9d8c0969fce74f9591a2fcb4a6da8506

load(d1)
mm = 1500;
cut = 0.05;
point = [0, 0; -1, 0];
ntime = size(SV,2);
ng = length(parm.gammas);
ne = length(parm.etas);
GD1 = find(parm.gammas==0);
GD2 = find(parm.etas==0);

nn = parm.Nepochs;
timevec = [1:10:nn-1,nn];
allmode = SV;
mod_sv = [1,2,4,8];
SV_s = SV(mod_sv,:,:); % top,mid,bottom

B = reshape(SV_s,[length(mod_sv),ntime,ne,ng]); % singular values
C = reshape(timecourse,[3,ntime,ne,ng]); % error, w1, w2 timecourses

Y = parm.data;
[U,S,V] = svd(Y,'econ');
targ_sv = diag(S); targ_sv = targ_sv(mod_sv);
item = 1;
feature = 5;
nc = size(V,2);
%a feature m for item i receives a contribution from each mode u_m vi 
coeffs = U(feature,1:nc).*V(item,:);

close all;
time = ntime;
npair = size(SV_s,3);


for pair = 1:npair
    
    err_f = timecourse(1,end,pair);
    end_err(:,pair) = err_f;
    
    y = squeeze(SV_s(:,:,pair))';
    th = targ_sv'*cut;
    tmp = abs(bsxfun(@minus, y, th));
    [~,id] = min(tmp);
    denominator = id(end);
    
    if isnan(err_f)
        out1(:,pair) = nan;
    else
        out1(:,pair) = mean(diff(id))/denominator;
    end
    
    error = timecourse(1,:,pair);
    h = error(1) - error(end);
    tmp = abs(error - error(end) - h*0.01);
    [~,id2] = min(tmp);
    tmp = abs(error - error(end) - h*0.99);
    [~,id1] = min(tmp);
    err = error(id1:id2);
    
    grad1 = diff(err);
    grad1 = grad1./max(abs(grad1));
    th = mean(abs(grad1)) + std(abs(grad1))*1.5;
    pks = 0;
    if ~isempty(grad1) && length(grad1)>1
        pks = findpeaks(-(grad1),'MinPeakHeight',0.001,'MinPeakWidth',0,'WidthReference','halfheight');
    else
        pks = 0;
    end
    out2(:,pair) = length(pks);
    if pks==0
        out2(:,pair) = nan;
    end
    
    % illusory correlation:
    IC = (coeffs(1,:)*allmode(:,:,pair))';
    out3(:,pair) = sum(IC);
   
    % 
    tmp = timecourse(2:3,:,pair);
    tmp = sum(tmp,2);
    W(:,pair) = (tmp(1)-tmp(2))./(tmp(1)+tmp(2));
    if (tmp(1)+tmp(2)) < 1
        W(:,pair) = nan;
    end
end

s = end_err;
s = reshape(s,[ne,ng]);
end_err = s;

s = out1;
s = reshape(s,[ne,ng]);
out1 = s;
disp('done')


s = out2;
s = reshape(s,[ne,ng]);
out2 = double(s)-1;
out2(out2>3)=3;

s = out3;
s = reshape(s,[ne,ng]);
out3 = s;

s = W;
s = reshape(s,[ne,ng]);
W = s;

figure('position',[280         586        1028         369])
subplot(1,3,1)
up = out1(GD2,GD1);
imagesc(out1,'AlphaData',double(~isnan(out1)),[0,1]*up*1.2);
hold on;
contour(out1,10,'color','k')
colorbar;
colormap(gca,viridis)
axis square
xlabel('Top-down feedback \gamma')
ylabel('Hebbian component \eta')
set(gca,'YDir','normal','fontsize',16)
set(gca,'color',0*[1 1 1]);
xid = [1,GD1,length(parm.gammas)];
yid = [1,GD2,length(parm.etas)];
set(gca,'xtick',xid,'xticklabel',parm.gammas(xid))
set(gca,'ytick',yid,'yticklabel',parm.etas(yid))
grid on

for cc = 1:size(point,1)
    [~,x] = min(abs(parm.gammas-point(cc,1)));
    [~,y] = min(abs(parm.etas-point(cc,2)));
    hold on;
    plot(x,y, 'r+', 'MarkerSize', 10, 'LineWidth', 2);
end

subplot(1,3,2)
imagesc(out2,'AlphaData',double(~isnan(out2)),[0,3]);
colormap(gca,gray)
colorbar;
axis square
xlabel('Top-down feedback \gamma')
set(gca,'YDir','normal','fontsize',16)
set(gca,'color',0*[1 1 1]);
set(gca,'xtick',xid,'xticklabel',parm.gammas(xid))
set(gca,'ytick',yid,'yticklabel',parm.etas(yid))
grid on
colorbar

for cc = 1:size(point,1)
    [~,x] = min(abs(parm.gammas-point(cc,1)));
    [~,y] = min(abs(parm.etas-point(cc,2)));
    hold on;
    plot(x,y, 'r+', 'MarkerSize', 10, 'LineWidth', 2);
end

subplot(1,3,3)
up = out3(GD2,GD1); % 58.8733
imagesc(out3,'AlphaData',double(~isnan(out3)),[0,up]);
hold on;
contour(out3,10,'color','k')
colorbar;
colormap(gca,viridis)
axis square
xlabel('Top-down feedback \gamma')
set(gca,'YDir','normal','fontsize',16)
set(gca,'color',0*[1 1 1]);
xid = [1,GD1,length(parm.gammas)];
yid = [1,GD2,length(parm.etas)];
set(gca,'xtick',xid,'xticklabel',parm.gammas(xid))
set(gca,'ytick',yid,'yticklabel',parm.etas(yid))
grid on


%% Supp Fig. 3: Frobenius norm of training error at the end of training
clc
maxerr = norm(Y,'fro')^1;
figure('position',[626   608   415   347]);
imagesc(sqrt(end_err),'AlphaData',double(~isnan(end_err)),[0,maxerr]);
hold on;
contour(W,'color','k')
colorbar;
colormap(gca,viridis)
axis square
xlabel('Top-down feedback \gamma')
ylabel('Hebbian component \eta')
set(gca,'YDir','normal','fontsize',16)
set(gca,'color',0*[1 1 1]);
set(gca,'xtick',xid,'xticklabel',parm.gammas(xid))
set(gca,'ytick',yid,'yticklabel',parm.etas(yid))
grid on
title('$$||\mathbf{y}-\mathbf{\hat{y}}||_{F}$$','Interpreter','Latex')


%% Figure 2d: Illustrative learning trajectories for two algorithms
clc;
fsize = 17;
point = [1,0; -1,0];

figure('position',[332         537        1025         228])
nrow = size(point,1);
np = 2;
for cc = 1:size(point,1)

    [~,x] = min(abs(parm.gammas-point(cc,1)));
    [~,y] = min(abs(parm.etas-point(cc,2)));

    mm = [1,3];
    subplot(1,4,mm(cc))
    ysig = B(:,:,y,x)';

    lc = linspace(0,0.8,size(ysig,2));
    for m = 1:size(ysig,2)
        plot(timevec,ysig(:,m),'linewidth',6-m,'color',ones(1,3)*lc(m));
        hold on;
    end

    th = targ_sv'*cut;
    tmp = abs(bsxfun(@minus, ysig, th));
    [~,id] = min(tmp);
    hold on;
    for i = 1:4
        plot(id(i)*10,ysig(id(i),i),'ko','markersize',8,'markerfacecolor','w');
        hold on
    end
    set(gca,'xtick','')
    set(gca,'xtick',[timevec(1),timevec(end)])
    ylim([0,4])
    if cc>1
        
    end

    if cc == 1
        ylabel('Strength of mode');
        xlabel('Time (Epochs)'); 
    end
    
    if cc == 1

    end
    set(gca,'fontsize',fsize,'box','off','linewidth',2)
    title(['\gamma = ',num2str(round(parm.gammas(x),1)),', \eta = ',num2str(round(parm.etas(y),1))])
    
    subplot(1,4,cc*2)
    plot(timevec,C(1,:,y,x)','linewidth',3,'color','k');
    ylim([0,40])
    
    if cc == 1
        ylabel('Sum squared error'); 
    end
    
    
    set(gca,'xtick',[timevec(1),timevec(end)])
    
    hold on
    error = C(1,:,y,x)';
    g = [0;diff(error)];
    g = g/max(abs(g(:)));
    yyaxis right
    plot(timevec,(g),'r','linewidth',1); ylim([-1,0.01])

    h = error(1) - error(end);
    tmp = abs(error - error(end) - h*0.05);
    [~,id2] = min(tmp);
    tmp = abs(error - error(end) - h*0.95);
    [~,id1] = min(tmp);
    err = error(id1:id2);
    grad1 = diff(err);
    grad1 = grad1./max(abs(grad1));
    th = mean(abs(grad1)) + std(abs(grad1))*1;
    out2(:,pair) = length(pks);
    set(gca,'fontsize',fsize,'box','off','linewidth',2)
end
