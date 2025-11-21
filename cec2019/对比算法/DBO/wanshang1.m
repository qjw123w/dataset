function [bestfit,Bestcost1,Bestcost2,pop,t]=wanshang1(fobj,fhd,lb,ub,dim,search_agent,FF)
%粪甲虫优化器参数初始化
np=search_agent;%种群大小

lb=lb(1);ub=ub(1);w=dim;maxit=500;%变量下界，上界，问题维数，算法最大迭代次数
FE=0;yz=1e-5;%初始评价次数，目标阈值精度
it=0;

bestsole.cost=inf;
bestsole.position=[];

RDBN_np= 0.2*np; % 滚球蜣螂个数
YDBN_np= 0.2*np; % 幼蜣螂个数
DDBN_np= 0.25*np; % 小蜣螂个数
TDBN_np=0.35*np; % 小偷蜣螂个数
%种群初始化并计算评价值
lb=lb.*ones(1,w);ub=ub.*ones(1,w);
for i=1:np
    popx(i,:)=lb+(ub-lb).*rand(1,w);
    pjpopx(i,:)=pjia(popx(i,:));
end
FE=FE+np;
ppopx=popx;pjppopx=pjpopx;%保存当前种群个体当前位置及评价值副本,将popx视为种群个体下一位置
ppopx1=ppopx;%保存种群个体上一位置
[fmin,BestIndex]=min(pjpopx);%找到当前最优个体信息及评价值
bestpopx=popx(BestIndex,:);%记录当前最优个体
trace(1,:)=fmin;%记录当前最优个体评价值
t=1;tic;
%算法迭代开始
while t<maxit
    [Worsepjpopx,WorseIndex]=max(pjpopx);%找到当前最差个体信息及评价值
    Worsepopx=popx(WorseIndex,:);%当前最差个体
    %滚球蜣螂位置更新模块，滚球蜣螂没有遇见障碍物会向后滚球，遇障碍物会跳舞重新选择方向滚球
    r1=rand;%跳舞或者滚球概率
    for i=1:RDBN_np
        if r1<0.9%没有遇见障碍物蜣螂滚球，滚球蜣螂位置更新如下
            %a选择
            r2=rand;%a选择概率
            if r2>0.5;a=1;else;a=-1;end
 %% 直接固定了怎么？
            k=0.1;b=0.3;           
            deltax=abs(ppopx(i,:)-Worsepopx);
            %滚球蜣螂位置为
            popx(i,:)=ppopx(i,:)+a*k*ppopx1(i,:)+b*deltax;
        else%滚球蜣螂遇见障碍物跳舞后更新位置
            zeta=randperm(180,1);%随机选择theta,theta在(0,pi]间
            %跳舞后滚球蜣螂位置更新为
            if zeta==0||zeta==90||zeta==180%此时蜣螂不更新
                popx(i,:)=ppopx(i,:);
            else
                theta=zeta*pi/180;
                popx(i,:)=ppopx(i,:)+tan(zeta)*abs(ppopx(i,:)-ppopx1(i,:));
            end
        end
        %对更新后的滚球蜣螂边界条件处理
        for j=1:w
            if popx(i,j)<lb(j)||popx(i,j)>ub(j)
                popx(i,j)=lb(j)+(ub(j)-lb(j))*rand;
            end
        end
        %更新滚球蜣螂评价值信息
        pjpopx(i,:)=pjia(popx(i,:));
        FE=FE+1;
 
    end
    %找到滚球蜣螂滚球or跳舞后滚球种群当前局部最佳位置
    [Bestp,BestI]=min(pjpopx);
    Bestpx=popx(BestI,:);%当前局部最佳位置
    %确定母蜣螂产卵区域
    R=1-t/maxit;
    lb1=max(Bestpx*(1-R),lb);ub1=min(Bestpx*(1+R),ub);%产卵区域下上界
    %幼蜣螂更新
    for i=(RDBN_np+1):(RDBN_np+YDBN_np)
        b1=rand(1,w);b2=rand(1,w);
        popx(i,:)=Bestpx+b1.*(ppopx(i,:)-lb1)+b2.*(ppopx(i,:)-ub1);
        for j=1:w%边界条件处理
            if popx(i,j)<lb(j)||popx(i,j)>ub(j)
               popx(i,j)=lb(j)+(ub(j)-lb(j))*rand;
            end
        end
        pjpopx(i,:)=pjia(popx(i,:));%更新幼蜣螂评价值
        FE=FE+1;
    end
    %找到此时最新的全局最佳位置
    [Bestp1,BestI]=min(pjpopx);
    Bestpx1=popx(BestI,:);%当前局部最佳位置
    %确定小蜣螂最佳觅食区域
    lb2=max(Bestpx1*(1-R),lb);ub2=min(Bestpx1*(1+R),ub);%小蜣螂最佳觅食区域下上界
    %最佳觅食区下上界得小蜣螂位置更新为
    for i=(RDBN_np+YDBN_np+1):(RDBN_np+YDBN_np+DDBN_np)
        c1=randn;c2=rand(1,w);
        popx(i,:)=ppopx(i,:)+c1.*(ppopx(i,:)-lb2)+c2.*(ppopx(i,:)-ub2);
        for j=1:w%边界条件处理
            if popx(i,j)<lb(j)||popx(i,j)>ub(j)
               popx(i,j)=lb(j)+(ub(j)-lb(j))*rand;
            end
        end
        pjpopx(i,:)=pjia(popx(i,:));%更新小蜣螂评价值
        FE=FE+1;
    end
    %找到此时最新的局部最佳位置
    [Bestp2,BestI]=min(pjpopx);
    Bestpx2=popx(BestI,:);%当前局部最佳位置
    %小偷蜣螂位置更新
    for i=(RDBN_np+YDBN_np+DDBN_np+1):np
        s=0.5;g=randn(1,w);
        popx(i,:)=bestpopx+s*g.*(abs(ppopx(i,:)-Bestpx2)+abs(ppopx(i,:)-bestpopx));
        for j=1:w%边界条件处理
            if popx(i,j)<lb(j)||popx(i,j)>ub(j)
               popx(i,j)=lb(j)+(ub(j)-lb(j))*rand;
            end
        end
        pjpopx(i,:)=pjia(popx(i,:));%更新小偷蜣螂评价值
        FE=FE+1;
    end
    %更新种群个体最优和全局最优
    ppopx1=ppopx;%更新种群个体上一位置
    for i=1:np%更新种群当前位置
        if pjpopx(i)<pjppopx(i)
           pjppopx(i)=pjpopx(i);
           ppopx(i,:)=popx(i,:);
        end
        if pjppopx(i)<fmin
            bestpopx=ppopx(i,:);
            fmin=pjppopx(i);
        end
    end
    trace(t+1)=fmin;
%     if trace(t+1)<yz;break;end
    t=t+1;
end
% toc
% plot(trace,'Color','r','LineWidth',2)
% title('函数优化求最值');xlabel('迭代次数');ylabel('适应度变化');
% grid on;box on;legend('DBO')
% disp(['运行时间: ', num2str(toc)]);
% disp(['最大评价次数: ', num2str(pjt)]);
% disp(['全局最佳解: ', num2str(bestpopx)]);
% disp(['全局最佳值: ', num2str(fmin)]);
% %优化问题函数
% function y=pjia(x)
%     y=sum(x.^2);%球面函数
% end
end