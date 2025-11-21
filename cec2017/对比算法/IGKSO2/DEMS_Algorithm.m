function  newPopulation= DEMS_Algorithm(elitePop, dim, population,popSize,F,numElite,ub,lb)
    % 参数说明:
    % fitnessFunc - 适应度函数
    % dim - 变量维度
    % lb - 变量下界
    % ub - 变量上界
    % maxIter - 最大迭代次数
    % popSize - 种群大小

    % 初始化种群
%     population = lb + (ub - lb) * rand(popSize, dim);
%     fitness = zeros(popSize, 1);
%     
%     % 计算初始适应度
%     for i = 1:popSize
%         fitness(i) = fitnessFunc(population(i, :));
%     end
    
%     % 记录最优解
%     [bestFitness, bestIdx] = min(fitness);
%     bestSolution = population(bestIdx, :);
%     
    % DEMS 主循环
%     for t = 1:maxIter
        % 计算当前 k 值（式 33）
        
        
        % 生成新解
        newPopulation = zeros(popSize, dim);
        for i = 1:popSize
            % 随机选择精英解 x_kbest
            kbestIdx = randperm(numElite,1);
            x_kbest = elitePop(kbestIdx, :);
            
            % 随机选择 4 个不同个体（式 34）
            candidates = randperm(popSize, 4);
            x_r1 = population(candidates(1), :);
            x_r2 = population(candidates(2), :);
            x_r3 = population(candidates(3), :);
            x_r4 = population(candidates(4), :);
            
            % 生成新解
            mutation = x_kbest + F * (x_r1 - x_r2) + F * (x_r3 - x_r4);
            newPopulation(i, :) = mutation;
            newPopulation(i, :) = max(min(newPopulation(i, :), ub), lb);
        end
%             
%             % 边界检查
%             newPopulation(i, :) = max(min(newPopulation(i, :), ub), lb);
%         end
%         
%         % 计算新适应度
%         newFitness = zeros(popSize, 1);
%         for i = 1:popSize
%             newFitness(i) = fitnessFunc(newPopulation(i, :));
%         end
%         
%         % 选择更优的解进入下一代
%         for i = 1:popSize
%             if newFitness(i) < fitness(i)
%                 population(i, :) = newPopulation(i, :);
%                 fitness(i) = newFitness(i);
%             end
%         end
%         
%         % 更新全局最优解
%         [currentBestFitness, currentBestIdx] = min(fitness);
%         if currentBestFitness < bestFitness
%             bestFitness = currentBestFitness;
%             bestSolution = population(currentBestIdx, :);
%         end
% 
%     end
end