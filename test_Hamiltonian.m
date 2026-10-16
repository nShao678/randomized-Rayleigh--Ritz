close all
rng(1)

n = 4000;
nn = n/2;
sigma = 1;
tau = 1e-3;
iterMax = 10;

err = ones(6,iterMax)*inf;
v0 = crandn(nn,1);
v0 = v0/norm(v0);
U = zeros(nn,iterMax);
P = crandn(nn,nn);
for iter = 1:iterMax
    U(:,iter) = expm(tau*iter*P)*v0;
end
Q1 = randunitary(nn);
Q2 = randunitary(nn);

Q = [Q1+Q2,Q1-Q2;Q1-Q2,Q1+Q2]/2;
lambda = sigma;
M = [zeros(nn),eye(nn);eye(nn),zeros(nn)];
A0 = crandn(nn,nn);
A0 = A0'+A0;
A0 = A0-v0*(v0'*A0)-(A0*v0)*v0'+v0'*A0*v0*(v0*v0');
A0 = (A0+A0')/2;

A2 = crandn(nn,nn);
A2 = A2'+A2;

U = [U;zeros(nn,iterMax)];
v = [v0;zeros(nn,1)];
v = Q*v;
U = Q*U;
for type = 1:2

    if type==1
        A1 = crandn(nn,nn);
        A1 = A1-(A1*v0)*v0'+(lambda*v0)*v0';
    else
        A1 = (lambda*v0)*v0';
    end
    A = [A0,-A1';A1,A2];




    A = Q*A*Q';

    for iter = 1:iterMax
        W = orth(U(:,1:iter));
        err(1,iter) = subspace(v,W);

        % RR
        [Y,Mu] = eig(W'*A*W,W'*M*W);
        [w,mu] = ritz(A,M,Y,Mu,W);
        err(2,iter) = subspace(w,v);
        err(3,iter) = abs(mu-lambda);

        % RRR
        Omega = randn(n,size(W,2))+1i*randn(n,size(W,2));
        [Y,Mu] = eig(Omega'*A*W,Omega'*M*W);
        [w,mu] = ritz(A,M,Y,Mu,W);
        err(4,iter) = subspace(w,v);
        err(5,iter) = abs(mu-lambda);
        err(6,iter) = abs((M*w)\(A*w)-lambda);

    end

    save(['data_ham',num2str(type)])
    xx = 1:10;
    figure
    hold on
    plot(xx,err(1,xx),'k-','LineWidth',2,'DisplayName','angle')
    plot(xx,err(2,xx),'r-','LineWidth',2,'DisplayName','RR-Vec')
    plot(xx,err(3,xx),'r--','LineWidth',2,'DisplayName','RR-Val')
    plot(xx,err(4,xx),'b-','LineWidth',2,'DisplayName','RRR-Vec')
    plot(xx,err(5,xx),'b--','LineWidth',2,'DisplayName','RRR-Val')
    plot(xx,err(6,xx),'b--o','LineWidth',2,'DisplayName','RRR-reVal')
    hold off
    set(gca,'yscale','log')
    set(gcf, 'Color', 'w');
    legend('box','off','FontSize',14)

    export_fig(['fig/ham',num2str(type),'.pdf'])
    export_fig(['fig/ham',num2str(type),'.eps'])
end
function [w,mu] = ritz(A,M,Y,Mu,W)
W = W*Y;
R = A*W-M*W*Mu;
[~,idx] = min(vecnorm(R));
mu = Mu(idx,idx);
w = W(:,idx);
end
function G = crandn(m,n)
G = (randn(m,n)+1i*randn(m,n));
end
function Q = randunitary(n)
[Q,R] = qr(crandn(n,n));
Q = Q*diag(sign(diag(R)));
end
