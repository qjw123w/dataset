% 1. 创建虚拟图形
fig_invisible = figure('Visible', 'off'); 
ax = axes(fig_invisible);

% 2. 绘制虚拟数据并添加图例
plot(ax, 0, 0, 'r', 'DisplayName', 'Line 1');
hold(ax, 'on');
plot(ax, 0, 0, 'b--', 'DisplayName', 'Line 2');
hLegend = legend(ax, 'show');
hLegend.Location = 'best';

% 3. 创建新图形并复制坐标区及图例
fig_visible = figure; 
new_ax = copyobj([ax, hLegend], fig_visible);

% 4. 隐藏坐标区，只显示图例
set(new_ax(1), 'Visible', 'off')  % 隐藏坐标区
set(fig_visible, 'Color', 'w', 'Name', '独立图例窗口');