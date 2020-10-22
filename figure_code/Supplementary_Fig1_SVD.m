
clear; clc; close all;
restoredefaultpath
Y = [1 1 1 1 1 1 1 1
     1 1 1 1 0 0 0 0
     0 0 0 0 1 1 1 1
     1 1 0 0 0 0 0 0
     0 0 1 1 0 0 0 0
     0 0 0 0 1 1 0 0
     0 0 0 0 0 0 1 1
     eye(4),zeros(4,4)
     zeros(4,4),eye(4)];
[U,S,V] = svd(Y,'econ');
X = eye(8);
Ex = X*X;
Eyx = Y*X';

close all;

fs = 16;
[u,s,v] = svd(Eyx,'econ');
v = v*s/max(diag(s));
% Adjust sign (arbitrary)
v = -v;
% v(:,[2 6 7 8]) = -v(:,[2 6 7 8]);
v(:,3:4) = v(:,[4,3]);
v(:,[2,4]) = -v(:,[2,4]);

% Reorder equal value vectors
v(:,5:end) = v(:,[8 5 6 7]);

% Clean up small sv vectors (numerical inaccuracies gum things up)
v(:,5:end) = (abs(v(:,5:end))>.1).*v(:,5:end);
v = bsxfun(@rdivide,v,sqrt(sum(v.^2)));
v(abs(v)<0.1) = 0;

u = u*s/max(diag(s));
u = -u;
u(:,[3,4]) = u(:,[4,3]);
u(:,5:8) = u(:,[8,5,6,7]);
u(:,[2,4]) = -u(:,[2,4]);
u(:,5:end) = (abs(u(:,5:end))>.1).*u(:,5:end);
u = bsxfun(@rdivide,u,sqrt(sum(u.^2)));

clc
figure('position',[510   695   560   260])
subplot(131)
imagesc(u,[-1,1]); 
set(gca,'ytick',1:15,'xtick',1:8,'xticklabel','')
set(gca,'fontsize',fs)
xlabel('Modes')
ylabel('Features')

subplot(132)
imagesc(s/max(diag(s)),[-1,1])
set(gca,'ytick',1:8,'xtick',1:8,'xticklabel','','yticklabel','')
axis square; 
h = text(3,1,'Modes','fontsize',fs);
set(h,'Rotation',-45);
set(gca,'fontsize',fs)


subplot(133)
imagesc(v',[-1,1])
set(gca,'ytick','','xtick',1:8)
axis square
colormap(redblue)
set(gca,'fontsize',fs)
xlabel('Objects')
ylabel('Modes')
