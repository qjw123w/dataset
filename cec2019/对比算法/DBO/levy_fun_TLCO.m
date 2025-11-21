function [z] = levy_fun_TLCO(n,m,beta)
% This function implements Levy's flight. 

% For more information see 
%'Multiobjective cuckoo search for design optimization Xin-She Yang, Suash Deb'. 

% Coded by Hemanth Manjunatha on Nov 13 2015.

% Input parameters
% n     -> Number of steps 
% m     -> Number of Dimensions 
% beta  -> Power law index  % Note: 1 < beta < 2

% Output 
% z     -> 'n' levy steps in 'm' dimension
%  n = 10;
%  m = 30; 
%  beta = 1.4;

    num = gamma(1+beta)*sin(pi*beta/3); % used for Numerator 
    
    den = gamma((1+beta)/3)*beta*2^((beta-1)/3); % used for Denominator

    sigma_u = (num/den)^(1/beta);% Standard deviation

    u = random('Normal',0,sigma_u^2,n,m); 
    
    v = random('Normal',0,1,n,m);

    z =( u./(abs(v).^(1/beta)));
    %z = zeros(n,m); n ;% n la so step, m la so bien
    
%     for i = 2:n % use those steps 
%     z(i,:) = z(i-1,:) + z1(i,:);    
%     end
%     
%     z = z(n,:);
%      
end



