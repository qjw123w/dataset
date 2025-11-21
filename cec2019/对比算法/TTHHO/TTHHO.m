

function [bestfit,Bestcost1,Bestcost2,pop,tk]=TTHHO(fobj,lb,ub,dim,N)
bestfit1=inf;

%Initialize the locations of Harris' hawks
X=initialization(N,dim,ub,lb);
MaxFES=100000;
% FES = 0;
FE=0;
% CNVG=zeros(1,T);
T=1000;
t=0; % Loop counter

while  FE < MaxFES
    t = t +1;
    Bestcost2(t)=bestfit1;
    for i=1:size(X,1)
        % Check boundries
        FU=X(i,:)>ub;FL=X(i,:)<lb;X(i,:)=(X(i,:).*(~(FU+FL)))+ub.*FU+lb.*FL;
        % fitness of locations
         fit(i,1)=feval(fobj,X(i,:));
        FE=FE+1;
        Bestcost1(FE)=bestfit1;
        % Update the location of Rabbit
        if  fit(i,1) <bestfit1
            bestfit1= fit(i,1) ;
            Rabbit_Location=X(i,:);
            best_voltage=Rabbit_Location;
        end
    end
    K=1;   % K is a real can be 0, 1, 2,....
    E1=2*(1-(t/T)); % factor to show the decreaing energy of rabbit
    % Update the location of Harris' hawks
    delta=rand()*(sin((pi/2)*(t/T))+cos((pi/2)*(t/T))-1);
    for i=1:size(X,1)
        r1=rand();r2=rand();
        r3 = rand();
        L=2*E1*r1-E1;
        C1=K*r2*E1+1;
        E0=2*rand()-1; %-1<E0<1
        Escaping_Energy=E1*(E0);  % escaping energy of rabbit
        r9=(2*pi)*rand();
        r10=2*rand;
        r11=rand();
        if abs(Escaping_Energy)>=1
            %% Exploration:
            % Harris' hawks perch randomly based on 2 strategy:
            
            q=rand();
            rand_Hawk_index = floor(N*rand()+1);
            X_rand = X(rand_Hawk_index, :);
            if q<0.5
                % perch based on other family members
                
                if r3<0.5
                    if r11<0.5
                        X(i,:)=X_rand-rand()*abs(X_rand-2*rand()*(best_voltage+exp(-L)*(X(i,:)+(E1*sin(r9)*abs(r10*best_voltage-X(i,:)))-C1*best_voltage)));
                    elseif r11>=0.5
                        X(i,:)=X_rand-rand()*abs(X_rand-2*rand()*(best_voltage+exp(-L)*(X(i,:)+(E1*cos(r9)*abs(r10*best_voltage-X(i,:)))-C1*best_voltage)));
                    end
                    
                    
                elseif r3>=0.5
                    if r11<0.5
                        X(i,:)=X_rand-rand()*abs(X_rand-2*rand()*(best_voltage+exp(-L)*(cos(L*2*pi)+tan(L*2*pi))*abs(X(i,:)+(E1*sin(r9)*abs(r10*best_voltage-X(i,:)))-C1*best_voltage)));
                    elseif r11 >= 0.5
                        X(i,:)=X_rand-rand()*abs(X_rand-2*rand()*(best_voltage+exp(-L)*(tan(L*2*pi)+sin(L*2*pi))*abs(X(i,:)+(E1*cos(r9)*abs(r10*best_voltage-X(i,:)))-C1*best_voltage)));
                    end
                    
                end
            elseif q>=0.5
                
                % perch on a random tall tree (random site inside group's home range)
                X(i,:)=(Rabbit_Location(1,:)-mean(X))-rand()*((ub-lb)*rand+lb);
            end
            
        elseif abs(Escaping_Energy)<1
            %% Exploitation:
            % Attacking the rabbit using 4 strategies regarding the behavior of the rabbit
            
            %% phase 1: surprise pounce (seven kills)
            % surprise pounce (seven kills): multiple, short rapid dives by different hawks
            
            r=rand(); % probablity of each event
            
            if r>=0.5 && abs(Escaping_Energy)<0.5 % Hard besiege
                
                if r3<0.5
                    if r11<0.5
                        X(i,:)=(Rabbit_Location)-Escaping_Energy*abs(Rabbit_Location-(best_voltage+exp(-L)*(X(i,:)+(E1*sin(r9)*abs(r10*best_voltage-X(i,:)))-C1*best_voltage)));
                    elseif r11>=0.5
                        X(i,:)=(Rabbit_Location)-Escaping_Energy*abs(Rabbit_Location-(best_voltage+exp(-L)*(X(i,:)+(E1*cos(r9)*abs(r10*best_voltage-X(i,:)))-C1*best_voltage)));
                    end
                    
                    
                elseif r3>=0.5
                    if r11<0.5
                        X(i,:)=(Rabbit_Location)-Escaping_Energy*abs(Rabbit_Location-(best_voltage+exp(-L)*(cos(L*2*pi)+tan(L*2*pi))*abs(X(i,:)+(E1*sin(r9)*abs(r10*best_voltage-X(i,:)))-C1*best_voltage)));
                    elseif r11 >= 0.5
                        X(i,:)=(Rabbit_Location)-Escaping_Energy*abs(Rabbit_Location-(best_voltage+exp(-L)*(tan(L*2*pi)+sin(L*2*pi))*abs(X(i,:)+(E1*cos(r9)*abs(r10*best_voltage-X(i,:)))-C1*best_voltage)));
                    end
                    
                end
                
            end
            
            if r>=0.5 && abs(Escaping_Energy)>=0.5  % Soft besiege
                Jump_strength=2*(1-rand()); % random jump strength of the rabbit
                
                if r3<0.5
                    if r11<0.5
                        X(i,:)=(Rabbit_Location-(best_voltage+exp(-L)*(X(i,:)+(E1*sin(r9)*abs(r10*best_voltage-X(i,:)))-C1*best_voltage)))-Escaping_Energy*abs(Jump_strength*Rabbit_Location-(best_voltage+exp(-L)*(X(i,:)+(E1*sin(r9)*abs(r10*best_voltage-X(i,:)))-C1*best_voltage)));
                    elseif r11>=0.5
                        X(i,:)=(Rabbit_Location-(best_voltage+exp(-L)*(X(i,:)+(E1*cos(r9)*abs(r10*best_voltage-X(i,:)))-C1*best_voltage)))-Escaping_Energy*abs(Jump_strength*Rabbit_Location-(best_voltage+exp(-L)*(X(i,:)+(E1*cos(r9)*abs(r10*best_voltage-X(i,:)))-C1*best_voltage)));
                    end
                    
                    
                elseif r3>=0.5
                    if r11<0.5
                        X(i,:)=(Rabbit_Location-(best_voltage+exp(-L)*(cos(L*2*pi)+tan(L*2*pi))*abs(X(i,:)+(E1*sin(r9)*abs(r10*best_voltage-X(i,:)))-C1*best_voltage)))-Escaping_Energy*abs(Jump_strength*Rabbit_Location-(best_voltage+exp(-L)*(cos(L*2*pi)+tan(L*2*pi))*abs(X(i,:)+(E1*sin(r9)*abs(r10*best_voltage-X(i,:)))-C1*best_voltage)));
                    elseif r11 >= 0.5
                        X(i,:)=(Rabbit_Location-(best_voltage+exp(-L)*(tan(L*2*pi)+sin(L*2*pi))*abs(X(i,:)+(E1*cos(r9)*abs(r10*best_voltage-X(i,:)))-C1*best_voltage)))-Escaping_Energy*abs(Jump_strength*Rabbit_Location-(best_voltage+exp(-L)*(tan(L*2*pi)+sin(L*2*pi))*abs(X(i,:)+(E1*cos(r9)*abs(r10*best_voltage-X(i,:)))-C1*best_voltage)));
                    end
                    
                end
            end
            
            %% phase 2: performing team rapid dives (leapfrog movements)
            if r<0.5 && abs(Escaping_Energy)>=0.5 % Soft besiege % rabbit try to escape by many zigzag deceptive motions
                
                Jump_strength=2*(1-rand());
                
                if r3<0.5
                    if r11<0.5
                        X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-(best_voltage+exp(-L)*(X(i,:)+(E1*sin(r9)*abs(r10*best_voltage-X(i,:)))-C1*best_voltage)));
                    elseif r11>=0.5
                        X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-(best_voltage+exp(-L)*(X(i,:)+(E1*cos(r9)*abs(r10*best_voltage-X(i,:)))-C1*best_voltage)));
                    end
                    
                    
                elseif r3>=0.5
                    if r11<0.5
                        X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-(best_voltage+exp(-L)*(cos(L*2*pi)+tan(L*2*pi))*abs(X(i,:)+(E1*sin(r9)*abs(r10*best_voltage-X(i,:)))-C1*best_voltage)));
                    elseif r11 >= 0.5
                        X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-(best_voltage+exp(-L)*(tan(L*2*pi)+sin(L*2*pi))*abs(X(i,:)+(E1*cos(r9)*abs(r10*best_voltage-X(i,:)))-C1*best_voltage)));
                    end
                    
                end
%                 fit(i,1) = feval(fhd,X(i,:)',fobj)-FF;
%                 FE=FE+1;
%                 if  fit(i,1) <bestfit1
%                     bestfit1= fit(i,1) ;
%                 end
%                 Bestcost1(FE)=bestfit1;
                fit_X1 = feval(fobj,X1);
                FE=FE+1;
                if   fit_X1 <bestfit1
                    bestfit1=  fit_X1 ;
                end
                Bestcost1(FE)=bestfit1;
                % Update the location of Rabbit
                
                if fit_X1<fit(i,1) % improved move?
                    X(i,:)=X1;
                else % hawks perform levy-based short rapid dives around the rabbit
                    
                    if r3<0.5
                        if r11<0.5
                            X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-(best_voltage+exp(-L)*(X(i,:)+(E1*sin(r9)*abs(r10*best_voltage-X(i,:)))-C1*best_voltage)))+rand(1,dim).*Levy(dim);
                        elseif r11>=0.5
                            X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-(best_voltage+exp(-L)*(X(i,:)+(E1*cos(r9)*abs(r10*best_voltage-X(i,:)))-C1*best_voltage)))+rand(1,dim).*Levy(dim);
                        end
                        
                        
                    elseif r3>=0.5
                        if r11<0.5
                            X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-(best_voltage+exp(-L)*(cos(L*2*pi)+tan(L*2*pi))*abs(X(i,:)+(E1*sin(r9)*abs(r10*best_voltage-X(i,:)))-C1*best_voltage)))+rand(1,dim).*Levy(dim);
                        elseif r11 >= 0.5
                            X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-(best_voltage+exp(-L)*(tan(L*2*pi)+sin(L*2*pi))*abs(X(i,:)+(E1*cos(r9)*abs(r10*best_voltage-X(i,:)))-C1*best_voltage)))+rand(1,dim).*Levy(dim);
                        end
                        
                    end
%                     fit_X2 = feval(fhd,X2',fobj)-FF;
                    fit_X2 = feval(fobj,X2);
                    FE=FE+1;
                    if   fit_X2 <bestfit1
                        bestfit1=  fit_X2 ;
                    end
                    Bestcost1(FE)=bestfit1;
                    if (fit_X2 <fit(i,1)) % improved move?
                        X(i,:)=X2;
                    end
                end
            end
            
            if r<0.5 && abs(Escaping_Energy)<0.5 % Hard besiege % rabbit try to escape by many zigzag deceptive motions
                % hawks try to decrease their average location with the rabbit
                Jump_strength=2*(1-rand());
                X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-mean(X));
                fit_X1 = feval(fobj,X1);
                FE=FE+1;
                if   fit_X1 <bestfit1
                    bestfit1=  fit_X1 ;
                end
                Bestcost1(FE)=bestfit1;
                
                if fit_X1<fit(i,1) % improved move?
                    X(i,:)=X1;
                else % Perform levy-based short rapid dives around the rabbit
                    X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-mean(X))+rand(1,dim).*Levy(dim);
                    fit_X2 = feval(fobj,X2);
                    FE=FE+1;
                    if   fit_X2 <bestfit1
                        bestfit1=  fit_X2 ;
                    end
                    Bestcost1(FE)=bestfit1;
                    if (fit_X2 <fit(i,1)) % improved move?
                        X(i,:)=X2;
                    end
                end
            end
            %%
        end
    end
    pop=X;
    bestfit=Bestcost1(FE);
    Bestcost1(MaxFES+1:FE)=[];
    %         maxit=500;
    Bestcost2(T+1:t-1)=[];
    tk=t-1;
    %     t=t+1;
    %     CNVG(t)=Rabbit_Energy;
    %    Print the progress every 100 iterations
    %    if mod(t,100)==0
    %        display(['At iteration ', num2str(t), ' the best fitness is ', num2str(Rabbit_Energy)]);
    %    end
end
end

% ___________________________________
function o=Levy(d)
beta=1.5;
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
u=randn(1,d)*sigma;v=randn(1,d);step=u./abs(v).^(1/beta);
o=step;
end


%微信公众号搜索：淘个代码，获取更多免费代�?
%禁止倒卖转售，违者必究！！！！！
%唯一官方店铺：https://mbd.pub/o/author-amqYmHBs/work
%代码清单：https://docs.qq.com/sheet/DU3NjYkF5TWdFUnpu
