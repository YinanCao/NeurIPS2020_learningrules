clear; clc;close all;
restoredefaultpath
d1 = 'initialW=small.mat';

load(d1)
mm = 1500;
size(allweight) % [st_w1_chl,st_w1_h,st_w2_chl];

close all; clc;
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
C = reshape(allweight,[4,ntime,ne,ng]); % error, w1, w2 timecourses

npair = size(SV_s,3);
for pair = 1:npair

     tmp = allweight(1:2,:,pair);
     tmp = sum(tmp,2);
     dem = sum(allweight(3,:,pair),2);
     W(:,pair) = (tmp(1)-tmp(2))./dem;
    if (dem) < .01
        W(:,pair) = nan;
    end
end

s = W;
s = reshape(s,[ne,ng]);
W = s;

xid = [1,GD1,length(parm.gammas)];
yid = [1,GD2,length(parm.etas)];

close all; clc
figure('position',[626   608   415   347]);
imagesc(W,'AlphaData',double(~isnan(W)),[-1,1]*1);
hold on;
contour(W,'color','k')
colorbar;
colormap(gca,redblue)
axis square
xlabel('Top-down feedback \gamma')
ylabel('Hebbian component \eta')
set(gca,'YDir','normal','fontsize',16)
set(gca,'color',0*[1 1 1]);
set(gca,'xtick',xid,'xticklabel',parm.gammas(xid))
set(gca,'ytick',yid,'yticklabel',parm.etas(yid))
grid on
     
