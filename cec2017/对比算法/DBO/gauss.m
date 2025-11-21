% 产生高斯分布的随机数列
% mu为均值，sigma为标准差，n为随机数个数
function r = gauss(mu, sigma, n)
    r = sqrt(-2 * log(rand(n,1))) .* cos(2 * pi * rand(n, 1));
    r = r * sigma + mu;
end