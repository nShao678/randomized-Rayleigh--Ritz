rng(1)
close all
clear all

n = 4096;
[coeffs,fun,F] = butterfly(n);
len = length(coeffs);
n = size(coeffs{1},1);


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
    norm(F(rho)*x)
    lambda = rho;
    v = x;
    err = zeros(8,iterMax);
    projA = cell(1,len);
    for iter = 1:iterMax

        W = orth(X(:,1:iter));
        err(1,iter) = subspace(v,W);

        % RR
        for ii = 1:len
            projA{ii} = W'*coeffs{ii}*W;
        end
        [Y,Mu] = polyeig(projA{:});
        [w,mu] = ritz(Y,Mu,lambda,W);
        err(2,iter) = abs(mu-lambda);
        err(3,iter) = subspace(v,w);



        % RRR

        Omega = randn(size(W))+1i*randn(size(W));
        for ii = 1:len
            projA{ii} = Omega'*coeffs{ii}*W;
        end
        [Y,Mu] = polyeig(projA{:});
        [w,mu] = ritz(Y,Mu,lambda,W);
        err(4,iter) = abs(mu-lambda);

        coef = zeros(1,len);
        for ii = 1:len
            coef(ii) = w'*coeffs{len+1-ii}*w;
        end
        mu = roots(coef);
        err(5,iter) = min(abs(mu-lambda));
        err(6,iter) = subspace(v,w);

%         coef = zeros(1,len);
%         for ii = 1:len
%             coef(ii) = X(:,iter)'*coeffs{len+1-ii}*X(:,iter);
%         end
%         mu = roots(coef);
%         err(7,iter) = min(abs(mu-lambda));
% 
%         err(8,iter) = subspace(v,X(:,iter));
    end
    % coef

    if type==1
        xx = 1:20;
    else
        xx = 1:25;
    end
    save(['data_bf',num2str(type)])
    figure
    hold on
    plot(xx,err(1,xx),'k-','LineWidth',2,'DisplayName','angle')
    plot(xx,err(3,xx),'r-','LineWidth',2,'DisplayName','RR-Vec')
    plot(xx,err(2,xx),'r--','LineWidth',2,'DisplayName','RR-Val')
    plot(xx,err(6,xx),'b-','LineWidth',2,'DisplayName','RRR-Vec')
    plot(xx,err(4,xx),'b--','LineWidth',2,'DisplayName','RRR-Val')
    plot(xx,err(5,xx),'b--o','LineWidth',2,'DisplayName','RRR-reVal')
%     plot(err(8,:),'m-','LineWidth',2,'DisplayName','resinvit-vec')
%     plot(err(7,:),'m--','LineWidth',2,'DisplayName','resinvit-val')
    
    hold off
    set(gca,'yscale','log')
    set(gcf, 'Color', 'w');
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