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
    p(t)=(1-(t/Max_iter)^0.25+abs(w(t+1))*((t/Max_iter)^0.25-(t/Max_iter)^3)); %数组回头检查一下 %数组回头检查一下
    %    p(t)=2*(1-(t/Max_iter)^(t/Max_iter));
    %     p(t)=(1/(1+1.5*exp((10*t/Max_iter)-5)))+0.1*abs(w(t+1));
    %     p(t)=1/(1+1.5*exp((10*t/Max_iter)-5))-0.1*rand;
    %       p(t)=exp((Max_iter-t)/Max_iter)-fit(i)/(sum(fit)+eps);
    beta_min=0.2;
    beta_max=1.2;
    beta=beta_min+( beta_max-beta_min)*(1-(t/Max_iter)^3)^2;
    alpha=abs(beta.*sin(3*pi/2+sin(3*pi*beta/2)));
    %         fit1=mean(fit);
    [fit1, fit_indice] = sort(fit, 'ascend');
    %     N2=20-round(0.1*N*t/Max_iter);
    N1=10-round(0.1*N*((t/Max_iter)^2));
    X1=X(fit_indice(1:15),:);
    %         X3=X(fit_indice(1:N1),:);
    %     sim=zeros(size(X1,1),1);
    XU=[];XU_fit=[];
    %     for i=1:size(X1,1)
    %         sumx=0;sum1=0;sum2=0;
    %         for j=1:dim
    %             sumx=sumx+(xbest(1,j)*X1(i,j));
    %             sum1=sum1+(xbest(1,j)^2);
    %             sum2=sum2+(X1(i,j)^2);
    %         end
    %         sim(i,1)=sumx/(sqrt(sum1)*sqrt(sum2));
    %     end
    %
    %     [sort_sim,sim_index]=sort(sim);
    %     sim=[];
    %     sim_indexs=sim_index(1:N1-(round(0.6*N1-0.2*N1*t/Max_iter)));
    %     remind= setdiff(1:15,sim_indexs);
    %     sim_indexs1=remind(1:round(0.6*N1-0.2*N1*t/Max_iter));
    %     X3=[X1(sim_indexs,:);X1( sim_indexs1,:)];
    %     fit3=[fit1(sim_indexs);fit1(sim_indexs1)];
    [X3, fit3,sim_selected_idx] = selectPopulation(X, fit, t, Max_iter, N1,dim);
    % 找出 X 中不属于 X3 的行的索引
    [~, idxxx] = ismember(X, X3, 'rows');
    non_zero_mask = (idxxx ~= 0);
    A_non_zero = idxxx(non_zero_mask);
    original_indices = find(non_zero_mask); % 记录非零元素的原始位置
    
    % 步骤2：找到非零元素中不相邻的重复值后索引
    [~, ~, ic] = unique(A_non_zero, 'stable');
    counts = accumarray(ic, 1);
    duplicate_values = find(counts > 1);
    
    back_indices_relative = [];
    for val = duplicate_values'
        idx = find(ic == val);
        if length(idx) >= 2
            % 取每组重复值的第二个及之后的索引（相对A_non_zero的位置）
            back_indices_relative = [back_indices_relative; idx(2:end)];
        end
    end
    
    % 步骤3：映射回原始数组的索引
    back_indices_original = original_indices(back_indices_relative);
    remain_indices = find(idxxx == 0);  % 找到所有未匹配的行索引
    remain_indices=[remain_indices;back_indices_original];
    % 提取剩余个体
    X2 = X(remain_indices, :);
    fit2 = fit(remain_indices);
    X=[X3;X2];
    fit=[fit3;fit2];
    
    %     remind= setdiff(1:15,sim_index(1:round(0.5*N1)));
    %     sim_indexs=sim_index(1:round(0.5*N1));
    %     X3=[X1(sim_indexs,:);X1(remind(1:N1-round(0.5*N1)),:)];
    %     fit3=[fit1(sim_indexs);fit1(remind(1:N1-round(0.5*N1)))];
%         XU=[XU;X3];
%         XU_fit=[XU_fit;fit3];
    for i=1:N
          if mod(t,100)==0
                 new_X(i,:)=(min(X)+max(X)-X(i,:));
               else
             
        %         if ismember(X(i,:), X3, 'rows')
        if i<=size(X3,1)
            %             lamda=2*rand(1,dim)-1;
            rk=randperm(N1,2);
            q=((max(fit)-fit(i))/(max(fit)-min(fit))).^((t/Max_iter)^2);   %3
            yita=1;
            if rand<=0.5
                betau=(rand(1,dim).*2).^(1/(1+yita));
            else
                betau=(1./(2-2.*rand(1,dim))).^(1/(1+yita));
            end
            xbest21= 0.5*((1-betau).*X1(1,:)+(1+betau).*X1(min(sim_selected_idx),:));
            xbest22= 0.5*((1+betau).*X1(1,:)+(1-betau).*X1(min(sim_selected_idx),:));
            if i==fit_indice(1)
                xbest2=X(i,:);
            else
                X0=[xbest21;xbest22;X(i,:)];
                xbest2=X0(randperm(3,1),:);
            end
            new_X(i,:)=xbest2+q.*(X3(rk(1),:)-X3(rk(2),:));  %220
%             for j=1:dim
%                 if new_X(i,j)>ub(:,j)
%                     new_X(i,j)=ub(:,j)-mod(new_X(i,j)-ub(:,j),ub(:,j)-lb(:,j));
%                 elseif new_X(i,j)<lb(:,j)
%                     new_X(i,j)=lb(:,j)+mod(lb(:,j)-new_X(i,j),ub(:,j)-lb(:,j));
%                 end
%             end
%             new_fit(i,1)=feval(fobj,new_X(i,:));
%             evaluation_count = evaluation_count + 1;
%             if new_fit(i,1)<fit(i,1)
%                 XU=[XU;new_X(i,:)];
%                 XU_fit=[XU_fit;new_fit(i)];
%                 [~,min_indexXU]=max( XU_fit);
%                 if size(XU,1)>10-round(0.1*N*(t/Max_iter)^2)
%                     XU(min_indexXU, :) = [];
%                     %                             XU(randi(size(XU, 1)), :) = [];
%                     XU_fit(min_indexXU) = [];
%                 end
%                 fit(i,1)=new_fit(i,1);
%                 X(i,:)=new_X(i,:);
%             end
        else
%             if mod(t,100)~=0
                % if rand<0.5
                r3=randperm(N,2);
                lamda=randsrc(1,1,[-1,1]);
                while i==r3(1)||i==r3(2)
                    r3=randperm(N,2);
                end
                %                                 rr=(2*rand(1,dim)-1).*0.25*(1-(t/Max_iter)^2);
                %                         rr=(0.1+0.8*(cos(pi*t/(2*Max_iter))));
                if t/Max_iter<0.5
                    rr=0.2+(1-t/Max_iter)^0.25;
                else
                    rr=0.2+0.3*(t/Max_iter)^2;
                end
                rr1=randperm(N1,1);
                z=zeros(1,dim);
                z1=randi([1,dim],1);
                %                                                 z1=round(0.8*dim-round((t/Max_iter)*0.5*dim));
                positions = randperm(dim,z1);
                z(positions) = 1;
                xpool=X(1:N1,:);
                X3_suiji=xpool(randperm(N1,1),:);
%                 if isempty(XU)
%                      X3_suiji=X3(rr1,:);
%                 else
%                     X3_suiji=XU(randperm(size(XU,1),1),:);
%                 end
                %                 if isequal(X(i,:),X3_suiji)&&ss==1
                %                          u3=setdiff(size(XU,1),X3_rand);
                %                          X3_rand=u3(randperm(length(u3),1));
                %                          X3_suiji=XU(X3_rand,:);
                %                 elseif isequal(X(i,:),X3_suiji)&&ss==0
                %                          u2=setdiff(1:N1,rr1);
                %                          rr1=u2(randperm(length(u2),1));
                %                          X3_suiji=X3(rr1,:);
                %                 end
                new_X(i,:)=X(i,:)+z.*lamda*p(t).^2.*(X(r3(1),:)-X(r3(2),:))+z.*(rr).*(X3_suiji-X(i,:));
           
%             for j=1:dim
%                 if new_X(i,j)>ub(:,j)
%                     new_X(i,j)=ub(:,j)-mod(new_X(i,j)-ub(:,j),ub(:,j)-lb(:,j));
%                 elseif new_X(i,j)<lb(:,j)
%                     new_X(i,j)=lb(:,j)+mod(lb(:,j)-new_X(i,j),ub(:,j)-lb(:,j));
%                 end
%             end
%             new_fit(i,1)=feval(fobj,new_X(i,:));
%             evaluation_count = evaluation_count + 1;
%             if new_fit(i,1)<fit(i,1)
%                 %                 XU=[XU;new_X(i,:)];
%                 %                 XU=[XU;new_X(i,:)];
%                 %                 XU_fit=[XU_fit;new_fit(i)];
%                 %                 [~,min_indexXU]=max( XU_fit);
%                 %                 if size(XU,1)>15-round(0.1*N*(t/Max_iter)^2)zz
%                 %                     XU(min_indexXU, :) = [];
%                 %                     %                             XU(randi(size(XU, 1)), :) = [];
%                 %                     XU_fit(min_indexXU) = [];
%                 %                 end
%                 fit(i,1)=new_fit(i,1);
%                 X(i,:)=new_X(i,:);
%             end
        end
          end
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
            %                 XU=[XU;new_X(i,:)];
            %                 XU=[XU;new_X(i,:)];
            %                 XU_fit=[XU_fit;new_fit(i)];
            %                 [~,min_indexXU]=max( XU_fit);
            %                 if size(XU,1)>15-round(0.1*N*(t/Max_iter)^2)zz
            %                     XU(min_indexXU, :) = [];
            %                     %                             XU(randi(size(XU, 1)), :) = [];
            %                     XU_fit(min_indexXU) = [];
            %                 end
            fit(i,1)=new_fit(i,1);
            X(i,:)=new_X(i,:);
        end
    end
      if(min(fit)<bestfit1k)
    [bestfit1k,best_index]=min(fit);
        xbest=X(best_index,:);
    end  
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
                Xk(1,j)=l2*(X(u(3),j)-Xr(1,j))+Xr(1,j);%l2等于0时，为50*1的一列数，l2为1时，为一个值，这么写就把一个值放在50*1的一列上，一列全是这个值
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
        Flag4ub= new_X(i,:)>ub;%对于二维数组中的第i行，判断数组中的每个元素是否大于给定的上限值ub,返回1或者0
        Flag4lb= new_X(i,:)<lb;
        new_X(i,:)=( new_X(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;%若是上边界与下边界都不符合，是在边界的基础上再加上种群
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