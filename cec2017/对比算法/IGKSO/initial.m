function [best_fit,curve,X]=initial(N,Max_iter,lb,ub,dim,fobj)
% ub=100*ones(1,dim);
% lb=-100*ones(1,dim)
%%初始化
% f=zeros(1,20);
%准反向学习机制
X=lb+(ub-lb).*rand(N,dim);
% cs=(ub+lb)/2;
% for i =1:N
%     for j=1:dim
%        mp=(ub(:,j)+lb(:,j))-X(i,j);
%        if mp>cs
%        X(i,:)=cs+rand*(mp-cs);
%        else
%         X(i,:)=cs+rand*(cs-mp); 
%        end
%     end
%       chebyshev混沌初始化
%             chebyshev=8;
%             Chebyshev=rand(N,dim);
%             for i=1:N
%                 for j=2:dim
%                    X(i,j)=cos(chebyshev.*acos(Chebyshev(i,j-1)));
%                 end
%             end
% a = 0.5; b=0.2;
% X=rand(N,dim);
% for i=1:N
% for j=2:dim
% X(i,j)=mod(X(i,j-1)+a-b/(2*pi)*sin(2*pi*X(i,j-1)),1);
% end
% end
%tent映射+小孔成像反向学习
% tent=0.5;   %tent混沌系数
% X=rand(N,dim);
% for i=1:N
% for j=2:dim
% if X(i,j-1)<tent
% X(i,j)=(X(i,j-1)/tent)+rand/N;
% elseif X(i,j-1)>=tent
% X(i,j)=(1-X(i,j-1))/(1-tent)+rand/N;
% end
% end
% end
% for i=1:N
% for j=1:dim
% X1(i,j)=X(i,j)*(ub(:,j)-lb(:,j))+lb(:,j);
% n=1;
% X(i,j)=(ub(:,j)+lb(:,j))/2+(ub(:,j)+lb(:,j))/(2*n)-X1(i,j)/n;
% end
% end

% end
%%计算适应度
for i=1:length(X)
    fit(i,1)=feval(fobj,X(i,:));
    %  fit(i,1)=feval(fhd,X(i,:)',func_num);
    %     fit(i,1)=fitness(X(i,:));
end
%%找到最优值

[best_fit,best_index]=min(fit);
xbest=X(best_index,:);


% fit1=sort(fit);  %对适应度进行排序
% pbest=X;
% fitpbest=fit;
m=1.5;
curve=zeros(Max_iter,1); %每一代的最优值基于代数画一个曲线
% p=zeros(500,1);
% w=zeros(500,1);
w(1)=0.1;
% m(1)=0.01;
%t=1;
% n1=zeros(4,50);
% count1=zeros(Max_iter,1);
% count2=zeros(Max_iter,1);
% count3=zeros(Max_iter,1);
% count4=zeros(Max_iter,1);

for t=1:Max_iter
    %while t<=T
    %     for i=1:size(X,1)
    %         %position_history(i,t,:)=X(i,:);
    %         %Trajectories(:,t)=X(:,1);
    %             fitness_history(i,t)=fobj(X(i,:));
    %     end
    %         a(t)=log(9/4)*t/Max_iter-log(0.9);
    %         w1(t)=exp(-a(t));
    %     % w1(t)=1-cosh((exp(t/Max_iter))/(exp(1)-1))^2;
    w(t+1)=1-2*(w(t))^4;
%       winitial=0.4;
%     wfinal=0.9;
%     w1=rand*(winitial-(winitial-wfinal)*(exp((t/Max_iter)-1))/(exp(1)-1));
    %     w1(t)=t/Max_iter;
    %     dn=ceil((N/4)*(cos((t/Max_iter)*pi)+1)+0.001);  %dn随迭代次数的增加而减小
    
    %     [m1,~]=find(fit<=fit1(dn),dn);%找到适应度小于dn得个体位置
    %%自适应权重系数
%     w3(t)=rand*(cosh(2*(1-(10*t/Max_iter))^4)-1);%(从小变大)
%       A=2-2*(10*exp(-6)/365)^(0.7*(Max_iter-t)/Max_iter)^2;
%       w3(t)=2*A^2*tan(rand*A/2);
    %     dnx=X(find(fit(1:dn)),:);
%      beta1=1.5+1/(1+exp(t/Max_iter));
%             m(t+1)=m(t)+0.025/(Max_iter*m(t));
    %p(t,1)=2*(1-(t/T)^0.25+abs(w(t+1))*((t/T)^0.25-(t/T)^3)); %数组回头检查一下
    
     p(t)=2*(1-(t/Max_iter)^0.25+abs(w(t+1))*((t/Max_iter)^0.25-(t/Max_iter)^3)); %数组回头检查一下 %数组回头检查一下
     
%    alpha1=0.5*cos(t*pi/250)+0.5;
%    alpha2=0.2*cos(t*pi*(-0.05))+0.2;

    %     p(t)=(1-(t/Max_iter))^(1-(t/Max_iter));
    %p(t,1)=2*(1-((t + eps)/T)^0.25+abs(w(t+1))*(((t + eps)/T)^0.25-((t + eps)/T)^3));
    %     p(t)=2*(1-(t/Max_iter)^(1/4))+abs(w(t+1))*((t/Max_iter)^(1/4)-((t/Max_iter)^3));
    beta_min=0.2;
    beta_max=1.2;
    beta=beta_min+( beta_max-beta_min)*(1-(t/Max_iter)^3)^2;
    alpha=abs(beta.*sin(3*pi/2+sin(3*pi*beta/2)));
    %% Hunting stage: Expand the search scope and search for potential prey positions
    for i=1:N
        for j=1:dim
            r1=rand;
            %                 r2=0.3+(1-(t/Max_iter)^3)^2;
            %                 r3=abs(r2*sin(1.5*pi+cos(3.5*pi*r2)));
            % r1=rand*2*pi;
            % r2=rand*pi;
            % tau=(sqrt(5)-1)/2;
            % theta1=-pi+2*pi*(1-tau);
            % theta2=-pi+2*pi*(tau);
            %             if t<Max_iter/2
            %             new_X(i,j)=X(i,j)+(lb(:,j)+r1*(ub(:,j)-lb(:,j)))/(t);
            %             else
            %                new_X(i,j)=X(i,j)+(lb(:,j)+r1*(ub(:,j)-lb(:,j)))/(0.5*t) ;
            %             end
            % 
             
%             new_X(i,j)=X(i,j)*abs(sin(r1))+r2*sin(r1)*((lb(:,j)+rand*(ub(:,j)-lb(:,j)))/(t))*levy(2)*abs(theta1*xbest(:,j)-theta2*X(i,j));
            new_X(i,j)=X(i,j)+(lb(:,j)+r1*(ub(:,j)-lb(:,j)))/t;
         
        end
        new_X(i,:) = max( new_X(i,:),lb);
        new_X(i,:) = min( new_X(i,:),ub);
        new_fit(i,1)= feval(fobj,new_X(i,:));
      
       
        if new_fit(i,1)<fit(i,1)
            fit(i,1)=new_fit(i,1);
            X(i,:)=new_X(i,:);
        end
    end
    %         count1(t,1)=length(find(n1(1,:)==1));
    %     if(min(fit2)<best_fit)
    %         [best_fit,best_index]=min(fit2);
    %         xbest=X(best_index,:);
    %     end
    
    %% Best Position Attraction Effect: Approaching the Best Position
    for i=1:N
        %%Moving towards the best hunting position (exploitation)
        %I(i,1)=feval(fhd,X(i,:)',func_num);
        %         I(i,1)=feval(fobj,X(i,:));
%         I(i,1)=(fit(i)-min(fit))/(max(fit)-min(fit)+eps);
 I(i,1)=fit(i);
%                         I=fit;
        
        %         w1(i)=(min(fit)-max(fit))/(fit(i,1)-max(fit));
        %          w1(i)=(1-exp(abs(min(fit))+abs(min(fit)))/exp(abs(fit(i))+abs(fit(i))))*0.6+(1-1/(1+exp(5*(Max_iter-2*t)/Max_iter)));
        s=m*(I(i,1).^rand);%s的值随r的改变而改变，即使是已经确定好的数也会随着下次另一个书的改变而整体改变
        %s=m*I.^r;  %eq4.3
        s=real(s);%返回实部
        %         q=randi([1,N]);
        %         u=ceil(rand(1,2)*N);
        %         if i==1
        %         p1max=0.2;
        %         p1min=0.8;
        %            theta=1+trnd(1)*tan(pi*(rand-0.5));
        % %         p1(i)=log(abs(min(fit))+abs(min(fit)))*(p1max-p1min)/log(abs(fit(i,1))+abs(fit(i,1)));
        %         if rand<0.8
        %             for j=1:dim
        %                 new_X(i,j)=w1(t)*X(i,j)+s*(rand*rand*xbest(:,j)-X(i,j));
        %             end
        %             % X(i,j)=s(i,1)*(xbest(:,j)-X(i,j));
        %         else
        %             if rand>0.9
        %             for j=1:dim
        %                 %Xu(i,j)=s(i,1)*(xbest(:,j)-X(i,j));
        %                 new_X(i,j)=theta*xbest(:,j)+s*(rand*rand*X(u(1),j)-X(u(2),j));  %与4.4所说的话对不上          eq4.4
        % %                                 new_X(i,j)=(Xj(i,j)+X(i-1,j))/2;
        %             end
        %             else
        %                 for j=1:dim
        %                 %Xu(i,j)=s(i,1)*(xbest(:,j)-X(i,j));
        %                 new_X(i,j)=w1(t)*xbest(:,j)+0.618*s*(rand*rand*X(u(1),j)-X(u(2),j));  %与4.4所说的话对不上          eq4.4
        % %                                 new_X(i,j)=(Xj(i,j)+X(i-1,j))/2;
        %             end
        %             end
        %         end
        %         if i==1
        %             for j=1:dim
        %                 new_X(i,j)=(1-w1(t))*(X(i,j)+(lb(:,j)+rand*(ub(:,j)-lb(:,j)))/t)+w1(t)*(s*(xbest(:,j)-X(i,j)));
        %             end
        % X(i,j)=s(i,1)*(xbest(:,j)-X(i,j));
        %         else
        %             for j=1:dim
        %                 %Xu(i,j)=s(i,1)*(xbest(:,j)-X(i,j));
        %                 Xj(i,j)=s*(xbest(:,j)-X(i,j));  %与4.4所说的话对不上          eq4.4
        %                 new_X(i,j)=(Xj(i,j)+X(i-1,j))/(2*rand);
        %             end
        %         end
        if i==1
            for j=1:dim
                new_X(i,j)=s*(xbest(:,j)-X(i,j));
                
            end
        else
            for j=1:dim
%                 D=(tan(pi*Max_iter/t)+1)/2;
                %Xu(i,j)=s(i,1)*(xbest(:,j)-X(i,j));
                Xj(i,j)=s*(xbest(:,j)-X(i,j));  %与4.4所说的话对不上          eq4.4
%                 new_X(i,j)=(D*xbest(:,j)+Xj(i,j))/2;
                  new_X(i,j)=(X(i-1,j)+Xj(i,j))/2;
            end
        end
%                for j=1:dim
%                         new_X(i,j)=s*(normrnd(0,1)*xbest(:,j)-X(i,j));
%                end
%         
        % Handling boundary violations
        
        if  new_X(i,:)>=lb & new_X(i,:)<=ub%
            new_fit(i,1)=feval(fobj,new_X(i,:));
            % %  new_fit(i,1)=feval(fhd,new_X(i,:)',func_num);
            %%比较适应度
            %                         if new_fit(i,1)<fit(i,1)
            %                             n1(2,i)=1;
            %                         elseif new_fit(i,1)>=fit(i,1)
            %                             n1(2,i)=0;
            %                         end
            
            if new_fit(i,1)<fit(i,1)
                fit(i,1)=new_fit(i,1);
                X(i,:)=new_X(i,:);
                
            end
        end
    end
    %         count2(t,1)=length(find(n1(2,:)==1));
    %fit3(i,1)=feval(fhd,X(i,:)',func_num);
    %         fit3(i,1)=feval(fobj,X(i,:));
    %           fit3(i,1)=fitness(X(i,:));
    
    %     if(min(fit3)<best_fit)
    %         [best_fit,best_index]=min(fit3);
    %         xbest=X(best_index,:);
    %     end
    %% Foraging stage: parabolic foraging
    
    for i=1:N
        lamda=randsrc(1,1,[-1,1]);%判断lamda是j运行10次一变还是j运行一次一变
        %         if rand>(1-t/Max_iter)
        
%         for j=1:dim
%            r2=rand;
%             %  X(i,j)=xbest(:,j)+r2*(xbest(:,j)-X(i,j))+lamda*(p(t)).^2*(xbest(:,j)-X(i,j));
%             %             new_X(i,j)=xbest(:,j)+r2*(xbest(:,j)-X(i,j))+lamda.*p(t).^2.*(xbest(:,j)-X(i,j));
% %             rr=randi([1,N]);
% %             new_X(i,j)=X(i,j)+levy1(1.5)*(X(rr,j)-X(i,j))+lamda.*p(t).^2.*(xbest(:,j)-X(i,j));
%              new_X(i,j)=X(i,j)+r2*(xbest(:,j)-X(i,j))+lamda.*p(t).^2.*(xbest(:,j)-X(i,j));
            
%         end
%         if t<=Max_iter/2
        for j=1:dim
            r2=rand;
%             z=pi/6+pi*(1-t/Max_iter)/3;
            %  X(i,j)=xbest(:,j)+r2*(xbest(:,j)-X(i,j))+lamda*(p(t)).^2*(xbest(:,j)-X(i,j));
            %             new_X(i,j)=xbest(:,j)+r2*(xbest(:,j)-X(i,j))+lamda.*p(t).^2.*(xbest(:,j)-X(i,j));
%             rr=randi([1,N]);
            new_X(i,j)=xbest(:,j)+r2*(xbest(:,j)-X(i,j))+lamda.*p(t).^2.*(xbest(:,j)-X(i,j));  
%               new_X(i,j)=X(i,j)+sin(z)*r2*(X(rr,j)-X(i,j))+lamda.*p(t).^2.*(xbest(:,j)-X(i,j));  
        end
%         else
%              for j=1:dim
%             r2=rand;
%             z=pi/6+pi*(1-t/Max_iter)/3;
%             new_X(i,j)=xbest(:,j)+cos(z)*r2*(xbest(:,j)-X(i,j))+lamda.*w1*p(t).^2.*(xbest(:,j)-X(i,j));  
% %             new_X(i,j)=xbest(:,j)+sin(z)*r2*(xbest(:,j)-X(i,j))+lamda.*p(t).^2.*(xbest(:,j)-X(i,j)); 
% %                l=-1+rand*2;
% %                  l=1-2*t/Max_iter;
% %           new_X(i,j)=abs(xbest(:,j)*2*rand-X(i,j))*exp(l)*cos(2*pi*l)+ xbest(:,j);   %这个地方直接跑，效果不好，尝试加惯性权重
% %         end
%              end

        %fit4(i,1)=feval(fhd,X(i,:)',func_num);
        %         fit4(i,1)=feval(fobj,X(i,:));
        %         fit4(i,1)=fitness(X(i,:));
        
        % Handling boundary violations
        new_X(i,:)=SpaceBound1( new_X(i,:),ub,lb);
        %评估适应度
        %     new_fit(i,1)=fitness(new_X(i,:));
        new_fit(i,1)=feval(fobj,new_X(i,:));
        if new_fit(i,1)<fit(i,1)
            fit(i,1)=new_fit(i,1);
            X(i,:)=new_X(i,:);
        end
        %         if fit(i) < fitpbest(i)
        %             pbest(i,:) = X(i,:);
        %             fitpbest(i,1) = fit(i,1);
        %         end
    end
    %% Self protection mechanism/dangerous escape behavior
    for i=1:N
        l1=randsrc(1,1,[0,1]);
        l2=randsrc(1,1,[0,1]);
        a1= l1*2*rand+(1-l1);
        a2= l1*rand+(1-l1);
        a3= l1*rand+(1-l1);
        rho=alpha.*(2*rand-1);
        %         q=randi([1,N]);
        %         a=1:50;
        %         a(best_index)=[];
        %         k=randi(length(a));
        %         u1=a(k);
        %         a(k)=[];
        %         u2=a(randi((length(a))));
        
        u=randperm(50,3);
            for j=1:dim
                X1(1,j)=lb(:,j)+rand*(ub(:,j)-lb(:,j));
                X2(1,j)=lb(:,j)+rand*(ub(:,j)-lb(:,j));
                Xr(1,j)=lb(:,j)+rand*(ub(:,j)-lb(:,j));%一组解
                Xk(i,j)=l2*(X(u(3),j)-Xr(1,j))+Xr(1,j);%l2等于0时，为50*1的一列数，l2为1时，为一个值，这么写就把一个值放在50*1的一列上，一列全是这个值
                k1=-1+2*rand();
                k2=randn();
                %        u1=ceil(rand(1,1)*dn);
                if a1<0.5
                    %  X(i,j)=X(i,j)+k1*(a1*xbest(:,j)-a2*Xk(i,j))+k2*rho*(a3*(X2(i,j)-X1(i,j)))+a2*(X(u1,j)-X(u2,j))/2;
                    %                 new_X(i,j)=X(i,j)+k1.*(a1*X(m1(u1),j)-a2*Xk(i,j))+k2*rho*(a3*(X2(1,j)-X1(1,j)))+a2*(X(u(1),j)-X(u(2),j))/2;
                    new_X(i,j)=X(i,j)+k1.*(a1*xbest(:,j)-a2*Xk(1,j))+k2*rho*(a3*(X2(1,j)-X1(1,j)))+a2*(X(u(1),j)-X(u(2),j))/2;
                else
                    % X(i,j)=xbest(:,j)+k1*(a1*xbest(:,j)-a2*Xk(i,j))+k2*rho*(a3*(X2(i,j)-X1(i,j)))+a2*(X(u1,j)-X(u2,j))/2;
                    %                 new_X(i,j)=xbest(:,j)+k1.*(a1*X(m1(u1),j)-a2*Xk(i,j))+k2*rho*(a3*(X2(1,j)-X1(1,j)))+a2*(X(u(1),j)-X(u(2),j))/2;
                    new_X(i,j)=xbest(:,j)+k1.*(a1*xbest(:,j)-a2*Xk(1,j))+k2*rho*(a3*(X2(1,j)-X1(1,j)))+a2*(X(u(1),j)-X(u(2),j))/2;
                end
            end

        % if rand <0.9
        %        new_X(i,j)=(1-w1(t))*(X(i,j)+k1.*(a1*xbest(:,j)-a2*Xk(i,j))+k2*rho*(a3*(X2(1,j)-X1(1,j)))+a2*(X(u(1),j)-X(u(2),j))/2)+w1(t)*(xbest(:,j)+k1.*(a1*xbest(:,j)-a2*Xk(i,j))+k2*rho*(a3*(X2(1,j)-X1(1,j)))+a2*(X(u(1),j)-X(u(2),j))/2);
        % else
        %     new_X(i,j)=min(X(:,j))+rand*(max(X(:,j))-min(X(:,j)));
        % end
        
        %         Check if solutions go outside the search space and bring them back
        Flag4ub= new_X(i,:)>ub;%对于二维数组中的第i行，判断数组中的每个元素是否大于给定的上限值ub,返回1或者0
        Flag4lb= new_X(i,:)<lb;
        new_X(i,:)=( new_X(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;%若是上边界与下边界都不符合，是在边界的基础上再加上种群
        
        
        %评估适应度
        %     new_fit(i,1)=fitness(new_X(i,:));
        new_fit(i,1)=feval(fobj,new_X(i,:));
        %  new_fit(i,1)=feval(fhd,new_X(i,:)',func_num);
        %                 if new_fit(i,1)<fit(i,1)
        %                     n1(4,i)=1;
        %                 elseif new_fit(i,1)>=fit(i,1)
        %                     n1(4,i)=0;
        %                 end
        %%比较适应度
        if new_fit(i,1)<fit(i,1)
            fit(i,1)=new_fit(i,1);
            X(i,:)=new_X(i,:);
        end
    end
    %         count4(t,1)=length(find(n1(4,:)==1));
    if(min(fit)<best_fit)
        [best_fit,best_index]=min(fit);
        xbest=X(best_index,:);
    end
        %%柯西随机反向扰动策略（不行）
%   xbest1=rand*(ub+lb)-xbest;
%   if randn<=0.5
%       xbest=rand*(xbest-xbest1);
%   else
%       xbest=trnd(1)*xbest;
%   end
    curve(t)=best_fit;
end
end




