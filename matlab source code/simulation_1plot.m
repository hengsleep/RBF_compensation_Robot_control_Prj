close all; % 关闭所有已有绘图窗口

figure(1); % 创建跟踪效果图窗口

subplot(211); % 创建2行1列第1个子图：位置跟踪
plot(t,x(:,1),'r',t,x(:,2),'k:','linewidth',2);
% x(:,1) = qd（期望位置，Sine Wave 直接来的 sin(t)）
% x(:,2) = q（实际位置，chap6_1plant 输出的第1维）
% x(:,3) = dq（实际速度，chap6_1plant 输出的第2维）
% 红线表示期望轨迹，黑色虚线表示系统跟踪结果

xlabel('time(s)'); % 横坐标：时间
ylabel('Position tracking'); % 纵坐标：位置跟踪
legend('ideal position','position tracking'); % 添加图例

subplot(212); % 创建2行1列第2个子图：速度跟踪
plot(t,cos(t),'r',t,x(:,3),'k:','linewidth',2);
% cos(t)：期望速度 dq_d
% x(:,3)：实际速度 dq
% 比较期望速度和实际速度跟踪效果

xlabel('time(s)');
ylabel('Speed tracking');
legend('ideal speed','speed tracking');

figure(2); % 创建扰动估计效果图窗口

plot(t,f(:,1),'r',t,f(:,2),'b','linewidth',2);
% f(:,1)：实际扰动 f
% f(:,2)：RBF网络估计扰动 fn
% 用于观察神经网络对未知扰动的逼近能力

xlabel('time(s)');
ylabel('f and fn');
legend('Practical uncertainties','Estimation uncertainties');