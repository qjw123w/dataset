clc
clear
tic
N=50;
Max_iter=500;
runs=20;
hanshu.dd=[];
ALA_huatu= repmat(hanshu, 10, 1);
for i =1:10
    F=i;
    for lun=1:runs
        [lb,ub,dim,fobj] = Get_Functions_cec2019(F);       
%         [best_fit,GKSO_curve,X]=cheng14(N,Max_iter,lb,ub,dim,fobj);
        [best_fit,GKSO_curve,X]=initial(N,Max_iter,lb,ub,dim,fobj);
%         plot(S);
%         [fMin , bestX,Convergence_curve ] = SSA(N, Max_iter,lb,ub,dim,fobj );
%                 bestsolution(lun)=best_fit;
%                 c10(lun,:)=GKSO_curve;%c10是20*500的一个数组
% %                 c11(lun,:)=GKSO_curve1;%c10是20*500的一个数组
        gfit=min(GKSO_curve);%cure表示每一代的最优值1*500的一个数组  gfit是500代中的一个最优值，是一个数
        f(lun,:)=gfit;  %f是20*1的一个数组
         hanshudd(lun,:)=GKSO_curve;
        diedaihui_ALA(i,:)=GKSO_curve;
%       gfit=min(Convergence_curve);%cure表示每一代的最优值1*500的一个数组  gfit是500代中的一个最优值，是一个数
%         f(lun,:)=gfit;  %f是20*1的一个数组
%   gfit1=min(GKSO_curve1);%cure表示每一代的最优值1*500的一个数组  gfit是500代中的一个最优值，是一个数
%         f1(lun,:)=gfit1;  %f是20*1的一个数组
    end
    ALA_huatu(i).dd=hanshudd;
%         for j=1:Max_iter
%             cc10(j)=mean(c10(:,j));%cc10 一代一代去求20轮的平均值
% %             cc11(j)=mean(c11(:,j));
%         end
    jilulunci(i,:)=f;  
    mean1=mean(f);
    average=abs(mean1-1);
    std1=sqrt(var(f));
%     best=min(f)-1;
%     worst=max(f)-1;
%     f-1
% % %     worst=max(f);
%     disp(['第F', num2str(F), '的平均值为：', num2str(average)]);
%     disp(['第F', num2str(F), '的标准差值为：', num2str(std1)]);
fprintf('%d       %d\n',average,std1);
%         disp(['第F', num2str(F), '的最差值值为：', num2str(worst)]);
%         for j=1:Max_iter/50
%             p(1)=1;
%             p(j+1)=j*50;
%         end
%         figure;
%         semilogy(p,cc10(p),'^-r','LineWidth',1.5,'MarkerSize',6);
% % hold on;
% %         semilogy(p,cc11(p),'^-b','LineWidth',1.5,'MarkerSize',6);
%         hold on
% %     legend('improved','initial');
%         set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',0.1);
%         title(['F',num2str(F)]);
%         xlabel('Iterations');
%         set(gca,'FontSize',14,'FontName','Times New Roman','LineWidth',2);
%         set(gca, 'LooseInset', [0,0,0,0]);  %为刻度标记留出一定的空间。若将其清零，则可以消除白边。
end
t1=toc 
% diedaihui_MTVSCA=huatu.pj;
save('ALA_huatu.mat','diedaihui_ALA')

