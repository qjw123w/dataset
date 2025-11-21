function output(data_MFGKSO,data_SOO_1,data_SOO_2)%data_SOO_1 和 data_SOO_2：单任务优化（DE）的结果数据（分别对应任务1和任务2）
Task1=[];
Task2=[];
for i=1:2:10
    Task1=[Task1,i];% 奇数索引（任务1的实验数据行）
    Task2=[Task2,i+1];% 偶数索引（任务2的实验数据行）
    %Task1 和 Task2 分别存储 data_MFPSO.EvBestFitness 中任务1和任务2的数据行索引。
end
x=[];
for i=1:1001
    x=[x,i];
end
%     taskName={'CI_HS','CI_MS','CI_LS','PI_HS','PI_MS','PI_LS','NI_HS','NI_MS','NI_LS'};
    taskName1={'CI\_HS','CI\_MS','CI\_LS','PI\_HS','PI\_MS','PI\_LS','NI\_HS','NI\_MS','NI\_LS'};
    taskNum=2;   %假设这里只处理三个任务组
%     last 数组按 [任务1代数, 任务2代数, 任务1代数, 任务2代数,...] 存储。
    %终止迭代次数，可能由于函数复杂度，造成终止迭代次数不同，表明9个实验，每个实验包括两个任务，公18个单独终止迭代次数
%     last=[600,1000,1000,1000,1000,1000,1000,300,1000,300,1000,1000,300,1000,600,1000,1000,1000];
%     last=1000*ones(1,18);
    for i=1:taskNum
%         start=1;
% %         step=50;
%         xstart=1;
        aveInd=1000;
        
        MFGKSO=data_MFGKSO(i);
        SOO_1=data_SOO_1(i);
        SOO_2=data_SOO_2(i);
        %这里taks1是1，3，5，evebestfitness里面的，而task2是2，4，6，。。。正好分开求平均值
        objTask1MFO(i,:)=mean(MFGKSO.EvBestFitness(Task1,:));
        objTask2MFO(i,:)=mean(MFGKSO.EvBestFitness(Task2,:));    %一行数1*迭代次数的数值
        objTask1SO(i,:)=mean(SOO_1.EvBestFitness);    %对轮数求均值，1*迭代次数的数值
        objTask2SO(i,:)=mean(SOO_2.EvBestFitness);
%         所求是最有任务1或任务2在迭代次数上适应度中最小的适应都
        %均值的最后一个代数   最小平均值任务1与任务2 
        aveTask1MFO(i)=objTask1MFO(aveInd);    %一个数
        aveTask2MFO(i)=objTask2MFO(aveInd);
        %迭代次数上的最小标准差，根据最后一列（min）适应度进行求标准差
        stdTaskMFO=MFGKSO.EvBestFitness(:,aveInd);
        %任务1或者任务2上的标准差    仅表示一个数值
        stdTask1MFO(i)=std(stdTaskMFO(Task1,:));
        stdTask2MFO(i)=std(stdTaskMFO(Task2,:));
        %         所求是单目标在任务1或任务2在迭代次数上适应度中最小的适应度
         %均值的最后一个代数   最小平均值任务1与任务2 单目标所求，不是多任务
        aveTask1SO(i)=min(objTask1SO);
        aveTask2SO(i)=min(objTask2SO);
        %迭代次数上适应度最小的一列，根据最后一列（min）适应度进行求标准差
        stdTask1SO(i)=std(SOO_1.EvBestFitness(:,aveInd/2));
        stdTask2SO(i)=std(SOO_2.EvBestFitness(:,aveInd/2));
        fprintf('第%d组任务的aveTask1MFO %f\n',i,aveTask1MFO(i));
        fprintf('第%d组任务的aveTask2MFO %f\n',i,aveTask2MFO(i));
        fprintf('第%d组任务的aveTask1SO% f\n',i,aveTask1SO(i));
        fprintf('第%d组任务的aveTask2SO %f\n',i,aveTask2SO(i));
        fprintf('第%d组任务的stdTask1MFO%f\n',i,stdTask1MFO(i));
        fprintf('第%d组任务的stdTask2MFO %f\n',i,stdTask2MFO(i));        
        fprintf('第%d组任务的stdTask1SO %f\n',i,stdTask1SO(i));
        fprintf('第%d组任务的stdTask2SO %f\n',i,stdTask2SO(i));    
        %%画图
figure(1)
% fname=i;
% d1=reshape(diedaihui_IIGKSOme(fname,1:100000),8,[]);
% D1=min(d1,[],1);
yyyy1=log(objTask1MFO(i,:));
yyyyy2=log( objTask1SO(i,:));
plot(yyyyy2,'Color',[0.1 0.8 0.9],'lineWidth',1.5)
hold on;
plot(yyyy1,'Color',[0.04 0.09 0.27],'lineWidth',1.5)
hod off
set(gca,'XTick',0:200:1000)
% %  set(gca,'XTicklabel',{'0','0.2','0.4','0.6','0.8','1.0'})
set(gca,'XLim',[0 1000])
set(gca,'XTicklabel',{'0','0.2','0.4','0.6','0.8','1.0'})
legend('MFIGKSO','GKSO');
xlabel('FEs(\times10^5)');
ylabel('factorial costs Value(log)');
% title('F1');
title(['%d: task1', num2str( taskName1(i))]);
%%%%
figure(2)
% fname=i;
% d1=reshape(diedaihui_IIGKSOme(fname,1:100000),8,[]);
% D1=min(d1,[],1);
yyyyy3=log(objTask2MFO(i,:));
yyyyy4=log( objTask2SO(i,:));
plot(yyyyy3,'Color',[0.42 0.35 0.80],'lineWidth',1.5)
hold on
plot(yyyyy4,'Color',[0.00 0.79 0.34],'lineWidth',1.5)
hold off
set(gca,'XTick',0:200:1000)
set(gca,'XLim',[0 1000])
set(gca,'XTicklabel',{'0','0.2','0.4','0.6','0.8','1.0'})
legend('MFIGKSO','GKSO');
xlabel('FEs(\times10^5)');
ylabel('factorial costs Value(log)');
title(['%d: task2', num2str( taskName1(i))]);
    end
end