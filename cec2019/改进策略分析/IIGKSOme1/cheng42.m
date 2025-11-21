function [bestfit,Bestcost1,Bestcost2,pop,tk]=cheng42(fobj,lb,ub,dim,N)
X=lb+(ub-lb).*rand(N,dim);
evaluation_count = 0;
bestfit1=inf;
MaxFES=100000;
% for i=N/2+1:N
%     X(i,:)=lb+rand(1,dim).*unifrnd(lb+(ub-lb)*2*(i-1)/N,lb+(ub-lb)*2*(i)/N);
% end
% ch(1)=0.01;
for i=1:N
    %     X(i,:)=dl(i)+rand(1,dim).*(du(i)-dl(i));
    fit(i,1)=feval(fobj,X(i,:));
    evaluation_count = evaluation_count + 1;
    if bestfit1> fit(i,1)
        bestfit1= fit(i,1);
    end
    Bestcost1(evaluation_count)=bestfit1;
    %     ch(i+1)=mod(ch(i)+0.2-0.5*sin(2*pi*ch(i))/(2*pi),1);
end
[bestfit1k,best_index]=min(fit);
xbest=X(best_index,:);
Max_iter=1000;
% curve=zeros(Max_iter,1); %每一代的最优值基于代数画一个曲线
w(1)=0.1;
A=[];
% t=1;
t=0;
while evaluation_count<100000
    % for t=1:Max_iter
    t = t +1;
    Bestcost2(t)=bestfit1;
    %     B=[];
    %     B1=[];
    w(t+1)=1-2*(w(t))^4;
    %         p(t)=1/(1+1.5*exp((10*t/Max_iter)-5))-0.1*rand;
    p(t)=2*(1-(t/Max_iter)^0.25+abs(w(t+1))*((t/Max_iter)^0.25-(t/Max_iter)^3)); %数组回头检查一下 %数组回头检查一下
    %    p(t)=2*(1-(t/Max_iter)^(t/Max_iter));
    %     p(t)=(1/(1+1.5*exp((10*t/Max_iter)-5)))+0.1*abs(w(t+1));
    %     p(t)=1/(1+1.5*exp((10*t/Max_iter)-5))-0.1*rand;
    %       p(t)=exp((Max_iter-t)/Max_iter)-fit(i)/(sum(fit)+eps);
    beta_min=0.2;
    beta_max=1.2;
    beta=beta_min+( beta_max-beta_min)*(1-(t/Max_iter)^3)^2;
    alpha=abs(beta.*sin(3*pi/2+sin(3*pi*beta/2)));
    
    if mod(t,100)==0
        for i=1:N
            new_X(i,:)=(min(X)+max(X)-X(i,:));
            for j=1:dim
                if new_X(i,j)>ub(:,j)
                    new_X(i,j)=ub(:,j)-mod(new_X(i,j)-ub(:,j),ub(:,j)-lb(:,j));
                elseif new_X(i,j)<lb(:,j)
                    new_X(i,j)=lb(:,j)+mod(lb(:,j)-new_X(i,j),ub(:,j)-lb(:,j));
                end
            end
            new_fit(i,1)=feval(fobj,new_X(i,:));
            evaluation_count = evaluation_count + 1;
            if bestfit1> new_fit(i,1)
                bestfit1= new_fit(i,1);
            end
            Bestcost1( evaluation_count)=bestfit1;
            if new_fit(i,1)<fit(i,1)
                fit(i,1)=new_fit(i,1);
                X(i,:)=new_X(i,:);
            end
        end
        if(min(fit)<bestfit1k)
            [bestfit1k,best_index]=min(fit);
            xbest=X(best_index,:);
        end
    end
    
    
    for i=1:N
        %%Moving towards the best hunting position (exploitation)
        %I(i,1)=feval(fhd,X(i,:)',func_num);
        %         I(i,1)=feval(fobj,X(i,:));
        %         I(i,1)=(fit(i)-min(fit))/(max(fit)-min(fit)+eps);
        I(i,1)=fit(i);
        %                         I=fit;
        
        %         w1(i)=(min(fit)-max(fit))/(fit(i,1)-max(fit));
        %          w1(i)=(1-exp(abs(min(fit))+abs(min(fit)))/exp(abs(fit(i))+abs(fit(i))))*0.6+(1-1/(1+exp(5*(Max_iter-2*t)/Max_iter)));
        s=1.5*(I(i,1).^rand);%s的值随r的改变而改变，即使是已经确定好的数也会随着下次另一个书的改变而整体改变
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
        new_X(i,:)=SpaceBound2( new_X(i,:),ub,lb);
            new_fit(i,1)=feval(fobj,new_X(i,:));
            evaluation_count = evaluation_count + 1;
            if bestfit1> new_fit(i,1)
                bestfit1= new_fit(i,1);
            end
            Bestcost1( evaluation_count)=bestfit1;
            
            if new_fit(i,1) < fit(i,1)    % 小于原有值就更新
                %                                                         A = [A; X(i,:)];
                %                                                         [~,min_indexA]=max(fit);
                %                                                         if size(A, 1) > 50
                % %                                                             A(randi(size(A, 1)), :) = [];    % 保持A的数目不超过popsize
                %                                                               A(min_indexA, :) = [];
                %                                                         end
                fit(i,1)=new_fit(i,1);
                X(i,:)=new_X(i,:);
                %             pre_X(i,:)=X(i,:);
                %                         memory_k2 = [memory_k2; k2];
                %                         if size(memory_k2, 1) > 50
                %                             memory_k2(randi(size(memory_k2, 1)), :) = [];    % 保持A的数目不超过popsize
                %                         end
            end
            
    end
    if(min(fit)<bestfit1k)
        [bestfit1k,best_index]=min(fit);
        xbest=X(best_index,:);
    end
    
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
        new_X(i,:)=SpaceBound2( new_X(i,:),ub,lb);
        new_fit(i,1)=feval(fobj,new_X(i,:));
        evaluation_count = evaluation_count + 1;
        if bestfit1> new_fit(i,1)
            bestfit1= new_fit(i,1);
        end
        Bestcost1( evaluation_count)=bestfit1;
        
        if new_fit(i,1) < fit(i,1)    % 小于原有值就更新
            %                                                         A = [A; X(i,:)];
            %                                                         [~,min_indexA]=max(fit);
            %                                                         if size(A, 1) > 50
            % %                                                             A(randi(size(A, 1)), :) = [];    % 保持A的数目不超过popsize
            %                                                               A(min_indexA, :) = [];
            %                                                         end
            fit(i,1)=new_fit(i,1);
            X(i,:)=new_X(i,:);
            %             pre_X(i,:)=X(i,:);
            %                         memory_k2 = [memory_k2; k2];
            %                         if size(memory_k2, 1) > 50
            %                             memory_k2(randi(size(memory_k2, 1)), :) = [];    % 保持A的数目不超过popsize
            %                         end
        end
        
    end
    if(min(fit)<bestfit1k)
        [bestfit1k,best_index]=min(fit);
        xbest=X(best_index,:);
    end
    
    U=0;
    for i=1:N
        
        %                                 PA=[X;A];
        %                                    r1=randi(size(PA,1));
        %         l1=randsrc(1,1,[0,1]);
        %         l2=randsrc(1,1,[0,1]);
        %         l3=randsrc(1,1,[-1,1]);
        l1 = randi([0,1]);
        l2 = randi([0,1]);
        l3 = randi([-1,1]);
        %                 a1= l1*2*(0.5+(1-cos(pi*t/(2*Max_iter)))/2)+(1-l1);
        
        %                 a3= l1*2*(0.5+(1-sin(pi*t/Max_iter))/2)+(1-l1);
        a1=l1*2*(0.5+sin(pi*t/Max_iter)/2)+(1-l1);
        a4=l1*2*(0.5+(1-sin(pi*t/Max_iter))/2)+(1-l1);
        a2= l1*(0.5+(1-cos(pi*t/(2*Max_iter)))/2)+(1-l1);
        a3= l1*(0.5+cos(pi*t/Max_iter)/2)+(1-l1);
        a5= l1*2*(0.5+(cos(pi*t/(Max_iter)))/2)+(1-l1);
        a6=l1*2*(0.5+sin(pi*t/Max_iter)/2)+(1-l1);
        %         a4=l1*2*(0.5+cos(pi*t/Max_iter)/2)+(1-l1);
        %         a4= l1*(0.5+sin(pi*t/Max_iter)/2)+(1-l1);
        %         a2= l1*(0.5+(cos(pi*t/(2*Max_iter)))/2)+(1-l1);
        %         a1= l1*2*(0.5+(1-cos(pi*t/(2*Max_iter)))/2)+(1-l1);
        %         a3= l1*2*(0.5+(cos(pi*t/(2*Max_iter)))/2)+(1-l1);
        %         a4= l1*(0.5+(1-cos(pi*t/(2*Max_iter)))/2)+(1-l1);
        rho=alpha.*(2.*rand-1);
        available = setdiff(1:N, i);
        u = available(randperm(length(available), 3));
        %         u1 = available(randperm(length(available), 1));
        %         u =randperm(N, 3);
        %         if i==u(3)
        %             candidates=setdiff(1:50,i);
        %             u(3)=randi(length(candidates));
        %         end
        k1=(-1+2*rand)*(1-(t/Max_iter)^0.25);
        k2=normrnd(0,1-(t/Max_iter)^2);
        while abs(k2)>1
            k2=normrnd(0,1-(t/Max_iter)^2);
        end
        %         Xk=l2*(X(i,:)-X(u1,:))+X(u1,:);
        Xk=l2*(X(i,:)-X(u(3),:))+X(u(3),:);
        %          Xk1=l2*(X(i,:)-Xs(aa1,:))+Xs(aa1,:);
        [~,fit_indexs]=sort(fit);
        Xs=X(fit_indexs,:);
        aa1=randperm(5,1);
        if isequal(Xk,Xs(aa1,:))
            l2=~l2;
            %             Xk=l2*(X(i,:)-X(u1,:))+X(u1,:);
            Xk=l2*(X(i,:)-X(u(3),:))+X(u(3),:);
        end
        X5=min(X)+rand(1,dim).*(max(X)-min(X));
        X6=min(X)+rand(1,dim).*(max(X)-min(X));
        %         P_ar= calculatePopulationDiversity(X);
        nVOL = calculateHypervolumeDiversity3(X, lb, ub);
        %                                              D = calculatePopulationDiversity(X);
        %                                             [h,h1] = calculatePopulationDiversity(X,ub,lb,xbest);% 计算多样性测度 h(k)
        %                                              Diver = calculatePopulationDiversity(X,ub,lb);
        for j=1:dim
            %             if rand<0.5
            if nVOL<0.15
                %               if P_ar<0.2*(t/Max_iter)^0.25
                new_X(i,j)=X(i,j)+(k1).*a1*(Xs(aa1,j)-Xk(1,j)) +a2*(X(u(1),j)-X(u(2),j))+k2*rho.*a3*( X5(:,j)- X6(:,j))*U;%+k2*rho.*a3*( X5(:,j)- X6(:,j))*U
                % %                 if abs(X(i,j) - new_X(i,j)) < 1e-10
                % %                     U=1;
                % %                 else
                % %                     U=0;
                % %                 end
            else
                % %                 %                 %                 %                if t/Max_iter<0.5
                % %                 %                 %                 f_values = fit - min(fit);
                % %                 %                 %                 F =  (1 - exp(-f_values / max(f_values + eps)));
                % %                 %                 %                 new_X(i,j)=X(i,j)+F(i).*(Xs(aa1,j)-X(i,j));
                % %                 %                 % %             else
                % %                 %                 % %                 new_X(i,j)=xbest(:,j)+exp(-t/Max_iter)*randn;
                % %                 %                 % %             end
                %                 new_X(i,j)=Xk(1,j)+(k1).*a6*(Xs(aa1,j)-Xk(1,j))+(a2)*(X(u(1),j)-X(u(2),j))+k2*rho.*a3*( X5(:,j)- X6(:,j))*U;
                new_X(i,j)=Xs(aa1,j)+(k1).*a4*(Xs(aa1,j)-Xk(1,j))+(a2)*(X(u(1),j)-X(u(2),j))/2+k2*rho.*a3*( X5(:,j)- X6(:,j))*U;
                % %                 %                   [new_X(i,:), ~] = gradient_local_search(xbest, min(fit), grad_obj, lb, ub, 1000, 1e-6);
            end
            if abs(X(i,j) - new_X(i,j)) < 1e-10
                U=1;
            else
                U=0;
            end
            if new_X(i,j)>ub(:,j)
                new_X(i,j)=ub(:,j)-min(new_X(i,j)-ub(:,j),ub(:,j)-lb(:,j)).*rand;
            elseif new_X(i,j)<lb(:,j)
                new_X(i,j)=lb(:,j)+min(lb(:,j)-new_X(i,j),ub(:,j)-lb(:,j)).*rand;
            end
            %                         if new_X(i,j)<lb(:,j)
            %                                         new_X(i,j)=min(X(:,j))+rand.*(max(X(:,j))-min(X(:,j)));
            %                             new_X(i,j)=lb(:,j)+(ub(:,j)-lb(:,j))*abs(new_X(i,j)-lb(:,j))/abs(new_X(i,j)-ub(:,j));
            %                         elseif new_X(i,j)>ub(:,j)
            %                             new_X(i,j)=lb(:,j)+(ub(:,j)-lb(:,j))*abs(new_X(i,j)-ub(:,j))/abs(new_X(i,j)-lb(:,j));
            %                         end
        end
        new_fit(i,1)=feval(fobj,new_X(i,:));
        evaluation_count = evaluation_count + 1;
        if bestfit1> new_fit(i,1)
            bestfit1= new_fit(i,1);
        end
        Bestcost1( evaluation_count)=bestfit1;
        
        if new_fit(i,1) < fit(i,1)    % 小于原有值就更新
            %                                                         A = [A; X(i,:)];
            %                                                         [~,min_indexA]=max(fit);
            %                                                         if size(A, 1) > 50
            % %                                                             A(randi(size(A, 1)), :) = [];    % 保持A的数目不超过popsize
            %                                                               A(min_indexA, :) = [];
            %                                                         end
            fit(i,1)=new_fit(i,1);
            X(i,:)=new_X(i,:);
            %             pre_X(i,:)=X(i,:);
            %                         memory_k2 = [memory_k2; k2];
            %                         if size(memory_k2, 1) > 50
            %                             memory_k2(randi(size(memory_k2, 1)), :) = [];    % 保持A的数目不超过popsize
            %                         end
        end
        
    end
    if(min(fit)<bestfit1k)
        [bestfit1k,best_index]=min(fit);
        xbest=X(best_index,:);
    end
    pop=X;
    bestfit=Bestcost1( evaluation_count);
    Bestcost1(MaxFES+1: evaluation_count)=[];
    Bestcost2(Max_iter+1:t-1)=[];
    tk=t-1;
end
end
% end