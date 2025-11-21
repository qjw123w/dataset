% Artificial lemming algorithm: A novel bionic meta-heuristic technique for solving real-world engineering optimization problems
%                                                                                                     
%  Developed in MATLAB R2024a                                                                 
%  Please follow wechat Swarm_Opti                                                                                                    
%  Author :Yaning Xiao, Hao Cui, Ruba Abu Khurma, Pedro A. Castillo                                           
%                                                                                                     
%         e-Mail: xiaoyn2023@mail.sustech.edu.cn; cuihao@nefu.edu.cn; Rubaabukhurma82@gmail.com; pacv@ugr.es                                                                                                                                                                                          
%_______________________________________________________________________________________________
% You can simply define your cost function in a seperate file and load its handle to fobj 
% The initial parameters that you need are:
%__________________________________________
% fobj = @YourCostFunction
% dim = number of your variables
% MaxFEs = number of function evaluations
% N = number of search agents
% lb=[lb1,lb2,...,lbn] where lbn is the lower bound of variable n
% ub=[ub1,ub2,...,ubn] where ubn is the upper bound of variable n
% If all the variables have equal lower bound you can just
% define lb and ub as two single numbers
% 2024-05-16
%______________________________________________________________________________________________
function [bestfit,Bestcost1,Bestcost2,pop,tk]=ALA(fobj,fhd,lb,ub,dim,N,FF)
tic
X=initialization(N,dim,ub,lb);% Initialize population
MaxFEs=300000;
% Position=zeros(1,dim); % Best position
% Score=inf; %Best score (initially infinite)
% fitness=zeros(1,size(X,1));% Fitness of each individual
% Convergence_curve=[];% Store convergence information
vec_flag=[1,-1]; % Directional flag
FEs=0;
 bestfit1=inf;
 Max_iter=6000;
%% Record the initial optimal solution and fitness
for i=1:size(X,1)  
%     FEs=FEs+1;
    fit(i,1) = feval(fhd,X(i,:)',fobj)-FF;
    FEs=FEs+1;
      if bestfit1> fit(i,1)
        bestfit1= fit(i,1);
        Position=X(i,:);
    end    
    Bestcost1(FEs)=bestfit1;
end

Iter=0; %Iteration number

%% Main optimization loop
while FEs<=MaxFEs
    Iter=Iter+1;
    Bestcost2( Iter)=bestfit1;
    RB=randn(N,dim);  % Brownian motion
    F=vec_flag(floor(2*rand()+1)); % Random directional flag
    theta=2*atan(1-FEs/MaxFEs); % Time-varying parameter
    for i=1:N
        E=2*log(1/rand)*theta;
        if E>1
           if rand<0.3
               r1 = 2 * rand(1,dim) - 1;
               Xnew(i,:)= Position+F.*RB(i,:).*(r1.*(Position-X(i,:))+(1-r1).*(X(i,:)-X(randi(N),:)));
          else
               r2 = rand ()* (1 + sin(0.5 * FEs));
               Xnew(i,:)= X(i,:)+ F.* r2*(Position-X(randi(N),:));
          end
        else
           if rand<0.5
               radius = sqrt(sum((Position-X(i, :)).^2));
               r3=rand();
               spiral=radius*(sin(2*pi*r3)+cos(2*pi*r3));
               Xnew(i,:) =Position + F.* X(i,:).*spiral*rand;
           else
               G=2*(sign(rand-0.5))*(1-FEs/MaxFEs);                     
               Xnew(i,:) = Position + F.* G*Levy(dim).* (Position - X(i,:)) ;
            end
        end
    end
    %%  Boundary check and evaluation
    for i=1:size(X,1)
        Flag4ub=Xnew(i,:)>ub;
        Flag4lb=Xnew(i,:)<lb;
        Xnew(i,:)=(Xnew(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb; % Boundary correction
%         newPopfit=fobj(Xnew(i,:)); % Evaluate new solution
          new_fit(i,1) = feval(fhd,Xnew(i,:)',fobj)-FF;
        FEs=FEs+1;
         if bestfit1> new_fit(i,1)
            bestfit1= new_fit(i,1);
            Position=X(i,:);
        end
         Bestcost1(FEs)=bestfit1;
            if new_fit(i,1)<fit(i,1)
                fit(i,1)=new_fit(i,1);
                X(i,:)= Xnew(i,:);
            end
    end
            pop=X;
        bestfit=Bestcost1(FEs);
        Bestcost1(MaxFEs+1:FEs)=[];
%         maxit=500;
        Bestcost2(Max_iter+1:Iter-1)=[];
        tk=Iter-1;
    %% Record convergence curve
%     Convergence_curve(Iter)=Score; 
%     Iter=Iter+1; 
end
end
function o=Levy(d)
beta=1.5;
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
u=randn(1,d)*sigma;v=randn(1,d);step=u./abs(v).^(1/beta);
o=step;
end