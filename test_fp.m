close all
rng(1)
n = 1000;
A = randn(n,n)+1i*randn(n,n);
B = randn(n,n)+1i*randn(n,n);
sigma = 0.01;
[V,Lambda] = eig(A,B,'vector');
[~,idx] =  min(abs(Lambda-sigma));
lambda = Lambda(idx);
v = V(:,idx);

iterMax = 10;
sampleMax = 2^17;

err = zeros(6,sampleMax);
m = 10;
p = 10;

W = randn(n,m)+1i*randn(n,m);


for iter = 1:iterMax
    W = orth((A-sigma*B)\(B*W));
end
tic
for sample = 1:sampleMax
if mod(sample,round(sampleMax/10))==0
    sample
    toc
end
Omega = randn(n,m)+1i*randn(n,m);
OmegaOs = randn(n,m+p)+1i*randn(n,m+p);

    % RRR
    [Y,Mu] = eig(Omega'*A*W,Omega'*B*W,'vector');
    [w,mu] = ritz(Y,Mu,lambda,W);
    err(1,sample) = abs(mu-lambda);
    err(2,sample) = subspace(v,w);
    err(3,sample) = abs(B*w\(A*w)-lambda);


    % RRR-oversampling
    [Uo,So,Vo] = svd([OmegaOs'*B*W,OmegaOs'*A*W]);
    [Y,Mu] = eig(Vo(m+1:2*m,1:m)',Vo(1:m,1:m)','vector');
    [w,mu] = ritz(Y,Mu,lambda,W);
    err(4,sample) = abs(mu-lambda);
    err(5,sample) = subspace(v,w);
    err(6,sample) = abs(B*w\(A*w)-lambda);


end
err0  =err;
for ii = 1:6
    err(ii,:) = sort(err(ii,:),'descend');
end

%%
figure
hold on
xx = round(linspace(1,sampleMax,50));
numdata = 100;
plot(xx,err(2,xx),'b-','LineWidth',2,'DisplayName','Ritz-vec')
plot(xx,err(1,xx),'r-','LineWidth',2,'DisplayName','Ritz-val')
plot(xx,err(3,xx),'m-','LineWidth',2,'DisplayName','Refined-val')
plot(xx,err(5,xx),'b--o','LineWidth',2,'DisplayName','Ritz-vec-os')
plot(xx,err(4,xx),'r--o','LineWidth',2,'DisplayName','Ritz-val-os')
plot(xx,err(6,xx),'m--o','LineWidth',2,'DisplayName','Refined-val-os')

hold off
set(gca,'YScale','log')
legend('Interpreter','latex','box','off','Location','southwest','FontSize',14,'NumColumns', 2)

set(gcf, 'Color', 'w');
export_fig(['fig/fp1.pdf'])
export_fig(['fig/fp1.eps'])

%%
xx = 2.^(1:8);
figure
hold on

p1 = plot(xx/sampleMax, err(2,xx), 'b-', 'LineWidth', 2, 'DisplayName', 'Ritz-vec');
p2 = plot(xx/sampleMax, err(1,xx), 'r-', 'LineWidth', 2, 'DisplayName', 'Ritz-val');
p3 = plot(xx/sampleMax, err(3,xx), 'm-', 'LineWidth', 2, 'DisplayName', 'Refined-val');

plot((xx/sampleMax), 1e-8./sqrt(xx/sampleMax), 'k--', 'LineWidth', 2);  

p5 = plot(xx/sampleMax, err(5,xx), 'b--o', 'LineWidth', 2, 'DisplayName', 'Ritz-vec-os');
p6 = plot(xx/sampleMax, err(4,xx), 'r--o', 'LineWidth', 2, 'DisplayName', 'Ritz-val-os');
p7 = plot(xx/sampleMax, err(6,xx), 'm--o', 'LineWidth', 2, 'DisplayName', 'Refined-val-os');

p8 = plot((xx/sampleMax), 2.5e-7./sqrt(xx/sampleMax), 'k--', 'LineWidth', 2, 'DisplayName', '$y\propto 1/\sqrt{x}$');

p_empty = plot(nan, nan, 'w', 'LineWidth', 0.1, 'HandleVisibility','on','DisplayName','');  

hold off
set(gca,'YScale','log')
set(gca,'XScale','log')

legend([p1 p2 p3 p_empty p5 p6 p7 p8], ...
       'Interpreter','latex', ...
       'box','off', ...
       'Location','east', ...
       'FontSize',14, ...
       'NumColumns',2)

set(gcf, 'Color', 'w');
export_fig(['fig/fp2.pdf'])
export_fig(['fig/fp2.eps'])


function [w,mu] = ritz(Y,Mu,lambda,W)
    [~,idx] = min(abs(Mu-lambda));
    mu = Mu(idx);
    w = W*Y(:,idx);
end