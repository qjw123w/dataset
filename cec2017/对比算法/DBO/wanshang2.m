function [bestfit,Bestcost1,Bestcost2,pop,t]=gai_rgdbo(fobj,fhd,lb,ub,dim,search_agent,FF)
%粪甲虫优化器参数初始�?
np=search_agent;%种群大小
lbxx=lb(1);ubsx=ub(1);%变量下界，上界，问题维数，算法最大迭代次�?
FE=0;maxFE=300000;
bestcost=inf;
bestsole.cost=inf;
bestsole.position=[];
it=0;maxit=2941;%2300;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
yz=1e-5;%初始评价次数，目标阈值精�?
RDBN_np= 0.2*np; % 滚球蜣螂个数
YDBN_np= 0.2*np; % 幼蜣螂个�?
DDBN_np=round(0.25*np) ; % 小蜣螂个�?
TDBN_np=np-RDBN_np-YDBN_np-DDBN_np;%0.35*np; % 小偷蜣螂个数
%% 拉丁超立方初始化种群%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%改进�?1
% step2:参数设置
% step3:划分小超立方�?
% lbld = (0:np-1)./np;
% ubld = (1:np)./np;;
% step4:产生�?个H*n的全排列矩阵
A = zeros(np, dim);
for i=1:dim
    A(:,i) = randperm(np);
end
% step5:采样
for ii1=1:np
    lbld(ii1)=lbxx+(ubsx-lbxx)/np*(ii1-1);
    ubld(ii1)=lbxx+(ubsx-lbxx)/np*(ii1);    
end
pop = zeros(np,dim);
for i=1:np
    for j=1:dim
        pop(i,j) = unifrnd(lbld(A(i,j)), ubld(A(i,j)));
    end
        fit_0(i,1)=feval(fhd,pop(i,:)',fobj)-FF;
        FE=FE+1;      
        if fit_0(i)<bestsole.cost
            bestsole.cost=fit_0(i);
            bestsole.position=pop(i,:);            
        end
        Bestcost1(FE)=bestsole.cost;
   
end
ppop=pop;ffit=fit_0;%%当前代是ppop
ppop1=ppop;%%上一�?
% [~,xu0]=sort(ffit);
bestsole.position=bestsole.position;

%% �?始循�?
while FE<maxFE
% while it<=maxit
    it=it+1;
    Bestcost2(it)=bestsole.cost;
    [~,xu1]=sort(ffit);
    xworst=ppop(xu1(np),:);
    for i=1:np
        if i<=RDBN_np
            if rand<0.9
                if rand>0.5
                    aef=1;
                else
                    aef=-1;
                end
                 k=0.1;b=0.3;  
%                 k=rand*(0.2-0)+0;b=rand*(1-0)+0;
                dertax=abs(ppop(i,:)-xworst);
                pop(i,:)=ppop(i,:)+aef*k*ppop1(i,:)+b*dertax;
            else
                cta=rand*(pi-0)+0;
                if cta==0||cta==pi/2||cta==pi
                    pop(i,:)=ppop(i,:);
                else
                    pop(i,:)=ppop(i,:)+tan(cta)*abs(ppop(i,:)-ppop1(i,:));
                end
            end
            for j=1:dim
                if pop(i,j)>ubsx||pop(i,j)<lbxx
                    pop(i,j)=lbxx+(ubsx-lbxx)*rand;
                end
            end
            fit_0(i,1)=feval(fhd,pop(i,:)',fobj)-FF;
            FE=FE+1;
            if fit_0(i)<bestsole.cost
                bestsole.cost=fit_0(i);
                bestsole.position=pop(i,:);
            end
            Bestcost1(FE)=bestsole.cost;
        end
        sulie=[1:np];
        sulie([RDBN_np+1:RDBN_np+YDBN_np])=[];
        jubu=fit_0;
        jubu([sulie],:)=[];
         [~,xu2]=sort(jubu);
        xju=pop(xu2(1)+RDBN_np,:);
%         [~,xu2]=sort(fit_0);
%         xju=pop(xu2(1),:);
        R=1-it/maxit;
        lb1=max(xju*(1-R),lb);ub1=min(xju*(1+R),ub);%产卵区域下上�?
        if i>RDBN_np&&i<=(RDBN_np+YDBN_np)
            b1=rand(1,dim);b2=rand(1,dim);
            pop(i,:)=xju+b1.*(ppop(i,:)-lb1)+b2.*(ppop(i,:)-ub1);
            for j=1:dim
                if pop(i,j)>ubsx||pop(i,j)<lbxx
                    pop(i,j)=lbxx+(ubsx-lbxx)*rand;
                end
            end
            fit_0(i,1)=feval(fhd,pop(i,:)',fobj)-FF;
            FE=FE+1;
            if fit_0(i,1)<bestsole.cost
                bestsole.cost=fit_0(i);
                bestsole.position=pop(i,:);
            end
            Bestcost1(FE)=bestsole.cost;
        end
        [~,xu3]=sort(fit_0);
        xqq=pop(xu3(1),:);
        lb2=max(xqq*(1-R),lb);ub2=min(xqq*(1+R),ub);
        ceshi1=min(lb2);ceshi2=max(ub2);
        if i>RDBN_np+YDBN_np&&i<=RDBN_np+YDBN_np+DDBN_np
            pop(i,:)=ppop(i,:)+randn*(ppop(i,:)-lb2)+rand(1,dim).*(ppop(i,:)-ub2);
             for j=1:dim
                if pop(i,j)>ubsx||pop(i,j)<lbxx
                    pop(i,j)=lbxx+(ubsx-lbxx)*rand;
                end
            end
            fit_0(i,1)=feval(fhd,pop(i,:)',fobj)-FF;
            FE=FE+1;
            if fit_0(i)<bestsole.cost
                bestsole.cost=fit_0(i);
                bestsole.position=pop(i,:);
            end
            Bestcost1(FE)=bestsole.cost;
        end
%         shulie=[1:RDBN_np;RDBN_np+YDBN_np+1:np];
        sulie=[1:np];
        sulie([RDBN_np+YDBN_np+DDBN_np+1:np])=[];
        jubu=fit_0;
        jubu([sulie],:)=[];
         [~,xu4]=sort(jubu);
        xju2=pop(xu4(1)+RDBN_np+YDBN_np+DDBN_np,:);
%         [~,xu4]=sort(fit_0);
%         xju2=pop(xu4(1),:);
        if i>RDBN_np+YDBN_np+DDBN_np
            s=0.5;g=randn(1,dim);
            pop(i,:)=bestsole.position+s*g.*(abs(ppop(i,:)-xju2)+abs(ppop(i,:)-bestsole.position));
            for j=1:dim
                if pop(i,j)>ubsx||pop(i,j)<lbxx
                    pop(i,j)=lbxx+(ubsx-lbxx)*rand;
                end
            end
            fit_0(i,1)=feval(fhd,pop(i,:)',fobj)-FF;
            FE=FE+1;
            if fit_0(i)<bestsole.cost
                bestsole.cost=fit_0(i);
                bestsole.position=pop(i,:);
            end
            Bestcost1(FE)=bestsole.cost;
        end
    end
ppop1=ppop;
    for i=1:np%更新种群当前位置
        if fit_0(i)<ffit(i)
            ffit(i)=fit_0(i);
            ppop(i,:)=pop(i,:);
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%改进�?2
%     [~,xu0]=sort(fit_0);
%     bestsole.position=bestsole.position;
    for i=1:np
        suiji=randperm(np);suiji(suiji==i)=[];
        aa1=suiji(1);aa2=suiji(2);
        xc1=(ppop(aa1,:)+ppop(aa2,:))/2;
        xc2=(ppop(aa1,:)+bestsole.position)/2;
        if it<maxit*(2/3)
            f=0.25;
            pop(i,:)=xc1+f*(xc1-ppop(i,:))+f*(xc2-ppop(i,:));
        else
            f=(1-2*rand)*0.5;
            pop(i,:)=bestsole.position+f*(xc1-ppop(i,:))+f*(xc2-ppop(i,:));
        end
        for j=1:dim
            if pop(i,j)>ubsx||pop(i,j)<lbxx
                pop(i,j)=lbxx+(ubsx-lbxx)*rand;
            end
        end
        fit_0(i,1)=feval(fhd,pop(i,:)',fobj)-FF;
        FE=FE+1;
        if fit_0(i)<bestsole.cost
            bestsole.cost=fit_0(i);
            bestsole.position=pop(i,:);
        end
        Bestcost1(FE)=bestsole.cost;
    end
    ppop1=ppop;
    for i=1:np%更新种群当前位置
        if fit_0(i)<ffit(i)
            ffit(i)=fit_0(i);
            ppop(i,:)=pop(i,:);
        end
       
    end
%     ppop1=ppop;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%改进�?3
    kk=(1+(it/maxit)^0.5)^10;
    xfan=(lbxx+ubsx)/2+(lbxx+ubsx)/(2*kk)-bestsole.position/kk;
    for j=1:dim
        if xfan(1,j)>ubsx||xfan(1,j)<lbxx
            xfan(1,j)=lbxx+(ubsx-lbxx)*rand;
        end
    end
    fitfan=feval(fhd,xfan',fobj)-FF;
    FE=FE+1;
    if fitfan<bestsole.cost
        bestsole.cost=fitfan;
        bestsole.position=xfan;
    end
        Bestcost1(FE)=bestsole.cost;
%     Bestcost1(FE)=bestsole.cost;
%     if fitfan<bestcost
    if fitfan>bestsole.cost
        bm=bestsole.position;
        ano=xfan;
        fbm=bestsole.cost;
    else
        bm=xfan;
        ano=bestsole.position;
        fbm=fitfan;
    end
    bmt=bm;
    for j=1:dim
        bm(1,j)=ano(1,j);
    end
        ffff=feval(fhd,bm',fobj)-FF;
        FE=FE+1;
        if ffff<bestsole.cost
            bestsole.cost=ffff;
            bestsole.position=bm;
        end
        Bestcost1(FE)=bestsole.cost;
%         Bestcost1(FE)=bestsole.cost;
        if ffff>fbm
            bm=bmt;
            bestsole.cost=fbm;
        else
            bestsole.cost=ffff; 
            bestsole.position=bm;
        end
   
    
  
end
it 
bestfit=bestsole.cost;
t=it;
Bestcost1(maxFE+1:size(Bestcost1,2))=[];



end