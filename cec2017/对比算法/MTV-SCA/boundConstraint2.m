%微信公众号搜索：淘个代码，获取更多免费代码
%禁止倒卖转售，违者必究！！！！！
%唯一官方店铺：https://mbd.pub/o/author-amqYmHBs/work
%代码清单：https://docs.qq.com/sheet/DU3NjYkF5TWdFUnpu
%%
function vi = boundConstraint2 (vi, lu)

% if the boundary constraint is violated, set the value to be the middle
% of the previous value and the bound
%
% Version: 1.1   Date: 11/20/2007
% Written by Jingqiao Zhang, jingqiao@gmail.com

[ps, D] = size(vi);  % the population size and the problem's dimension

vi=((vi>=lu(1, :))&(vi<=lu(2, :))).*vi...
        +(vi<lu(1, :)).*(lu(1, :)+0.25.*(lu(2, :)-lu(1, :)).*rand(ps,D))+(vi>lu(2, :)).*(lu(2, :)-0.25.*(lu(2, :)-lu(1, :)).*rand(ps,D));
%微信公众号搜索：淘个代码，获取更多免费代码
%禁止倒卖转售，违者必究！！！！！
%唯一官方店铺：https://mbd.pub/o/author-amqYmHBs/work
%代码清单：https://docs.qq.com/sheet/DU3NjYkF5TWdFUnpu
