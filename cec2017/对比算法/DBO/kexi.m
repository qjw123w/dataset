function y = Cauchy_rand(m,b,dim)
u = rand(1,dim);
y = m - (b ./ tan(pi .* u));
end