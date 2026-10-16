rng(1)
close all
clear all

n = 4096;
[coeffs,~,F] = butterfly(n);
len = length(coeffs);
n = size(coeffs{1},1);
nTrial = 1000;
qs = [0.1 0.5 0.9];

for type = 1:2
    iterMax = 100;
    if type==1
        sigma = 2i;
    else
        sigma = 1+1i;
    end
    DT = F(sigma);
    x = randn(n,1)+1i*randn(n,1);
    x = x/norm(x);
    X = zeros(n,iterMax);
    X(:,1) = x;
    for ii =1:iterMax
        [rho,x] = resinvit(x,DT,coeffs,sigma);
        X(:,ii+1) = x;
        if norm(F(rho)*x)<1e-12
            iterMax = ii-1;
            break
        end
    end
    lambda = rho;
    v = x;

    errRR = zeros(3,iterMax);
    errRRR = zeros(3,iterMax,nTrial);
    projA = cell(1,len);
    for iter = 1:iterMax
        W = orth(X(:,1:iter));
        errRR(1,iter) = subspace(v,W);

        for ii = 1:len
            projA{ii} = W'*coeffs{ii}*W;
        end
        [Y,Mu] = polyeig(projA{:});
        [w,mu] = ritz(Y,Mu,lambda,W);
        errRR(2,iter) = abs(mu-lambda);
        errRR(3,iter) = subspace(v,w);

        AW = cell(1,len);
        for ii = 1:len
            AW{ii} = coeffs{ii}*W;
        end

        for t = 1:nTrial
            Omega = randn(size(W))+1i*randn(size(W));
            for ii = 1:len
                projA{ii} = Omega'*AW{ii};
            end
            [Y,Mu] = polyeig(projA{:});
            [w,mu] = ritz(Y,Mu,lambda,W);
            errRRR(1,iter,t) = abs(mu-lambda);

            coef = zeros(1,len);
            for ii = 1:len
                coef(ii) = w'*coeffs{len+1-ii}*w;
            end
            mu = roots(coef);
            errRRR(2,iter,t) = min(abs(mu-lambda));
            errRRR(3,iter,t) = subspace(v,w);
        end
    end

    Q = quantile(errRRR,qs,3);

    if type==1
        xx = 1:20;
    else
        xx = 1:25;
    end
    figure
    hold on
    band = @(r) fill([xx fliplr(xx)],[Q(r,xx,1) fliplr(Q(r,xx,3))],'b', ...
        'FaceAlpha',0.15,'EdgeColor','none','HandleVisibility','off');
    band(3); band(1); band(2);
    plot(xx,errRR(1,xx),'k-','LineWidth',2,'DisplayName','angle')
    plot(xx,errRR(3,xx),'r-','LineWidth',2,'DisplayName','RR-Vec')
    plot(xx,errRR(2,xx),'r--','LineWidth',2,'DisplayName','RR-Val')
    plot(xx,Q(3,xx,2),'b-','LineWidth',2,'DisplayName','RRR-Vec')
    plot(xx,Q(1,xx,2),'b--','LineWidth',2,'DisplayName','RRR-Val')
    plot(xx,Q(2,xx,2),'b--o','LineWidth',2,'DisplayName','RRR-reVal')
    hold off
    set(gca,'yscale','log')
    set(gcf,'Color','w');
    legend('box','off','FontSize',14,'Location','southwest')
    export_fig(['fig/butterfly',num2str(type),'.pdf'])
    export_fig(['fig/butterfly',num2str(type),'.eps'])
end

function [w,mu] = ritz(Y,Mu,lambda,W)
[~,idx] = min(abs(Mu-lambda));
mu = Mu(idx);
w = W*Y(:,idx);
end

function [rho,x] = resinvit(x,DT,coeffs,sigma)
len = length(coeffs);
coef = zeros(len,1);
for ii = 1:len
    coef(ii) = x'*(coeffs{len+1-ii}*x);
end
rho = roots(coef);
[~,idx] = min(abs(rho-sigma));
rho = rho(idx);
y = coeffs{len}*x;
for ii = len-1:-1:1
    y = rho*y+coeffs{ii}*x;
end
x = x-DT\y;
x = x/norm(x);
end
