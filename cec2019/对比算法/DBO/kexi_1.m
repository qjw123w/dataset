function kexi_2=kexi_1(x1,sigama1)
weizhicanshu=x1;
sifenwei=sigama1*2;
pd = makedist('tLocationScale','mu',weizhicanshu,'sigma',sifenwei,'nu',1);
med = median(pd);
r = iqr(pd);
kexi_2=pd;
end
% clc
% clear
% % openExample('stats/GenerateCauchyRandomNumbersUsingStudentstExample')
% % rng('default');  % For reproducibility
% % r = trnd(1,10,1);
% pd = makedist('tLocationScale','mu',0.65,'sigma',0.1,'nu',1);
% med = median(pd);
% r = iqr(pd);
% %%柯西分布的中位数等于其位置参数，四分位数间距等于其尺度参数的两倍。其均值和标准差未定义。
% x = -20:1:20;
% y = pdf(pd,x);
% plot(x,y,'LineWidth',2)
% rng('default');  % For reproducibility
% r = random(pd,10,1);
% r = random(pd,5,5);