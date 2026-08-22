function [sys,x0,str,ts] = spacemodel(t,x,u,flag)
switch flag
case 0
    [sys,x0,str,ts]=mdlInitializeSizes; % 初始化Simulink模块参数、状态和输出
case 1
    sys=mdlDerivatives(t,x,u); % 计算连续状态导数（这里为RBF权值更新律）
case 3
    sys=mdlOutputs(t,x,u); % 计算控制输入和系统输出
case {2,4,9}
    sys=[]; % 未使用的flag类型
otherwise
    error(['Unhandled flag = ',num2str(flag)]);
end

function [sys,x0,str,ts]=mdlInitializeSizes
global c b m kv kp % 定义全局变量，供其他函数调用

m=5; % RBF神经网络隐藏层节点数量

sizes = simsizes;
sizes.NumContStates  = m; % 连续状态数量，即RBF网络权值数量
sizes.NumDiscStates  = 0; % 无离散状态
sizes.NumOutputs     = 3; % 输出：[控制力矩,真实扰动,RBF估计扰动]
sizes.NumInputs      = 3; % 输入：[期望位置,实际位置,实际速度]
sizes.DirFeedthrough = 1; % 输出直接依赖输入
sizes.NumSampleTimes = 1; % 一个采样时间
sys = simsizes(sizes);
x0  = zeros(1,m); % RBF初始权值设为0
str = [];
ts  = [0 0]; % 连续采样时间设置

if m==5
c=1/2*[-2 -1 0 1 2; % RBF中心点，二维输入空间(q误差和速度误差)
       -2 -1 0 1 2];
elseif m==19
c=1/9*[-9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9; % 19个RBF中心
       -9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9];
end
b=5; % 高斯函数宽度

alfa=3; % 误差动态参数
kp=alfa^2; % 比例控制参数
kv=2*alfa; % 速度反馈参数

function sys=mdlDerivatives(t,x,u) % 自适应律计算（更新RBF权值）
global c b m kv kp
qd=u(1); % 期望位置
dqd=cos(t); % 期望速度
ddqd=-sin(t); % 期望加速度

q=u(2); % 实际位置
dq=u(3); % 实际速度

e=q-qd; % 位置误差
de=dq-dqd; % 速度误差

A=[0 1;-kp -kv]; % 误差系统矩阵
D=10; % 名义惯量参数
B=[0 1/D]'; % 输入矩阵

Q=50*eye(2); % Lyapunov矩阵设计参数
P=lyap(A',Q); % 求解Lyapunov方程
eig(P); % 检查P矩阵特征值

if m==5
th=[x(1) x(2) x(3) x(4) x(5)]'; % 当前RBF权值
elseif m==19
th=[x(1) x(2) x(3) x(4) x(5) x(6) x(7) x(8) x(9) x(10) x(11) x(12) x(13) x(14) x(15) x(16) x(17) x(18) x(19)]'; % 当前19个权值
end

xi=[e;de]; % RBF输入向量：误差和误差变化率

h=zeros(m,1); % 初始化RBF基函数输出
for j=1:1:m
    h(j)=exp(-norm(xi-c(:,j))^2/(2*b^2)); % 高斯径向基函数
end
gama=200; % 自适应学习率

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S1=2; % 选择自适应律类型
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if S1==1       % 第一种自适应律
    S=gama*h*xi'*P*B; % 基于Lyapunov稳定性的权值更新
elseif S1==2    % 第二种带UUB项的自适应律
    k1=0.001; % 鲁棒修正系数
    S=gama*h*xi'*P*B+k1*gama*norm(xi)*th; % 增加权值约束项
end

S=S'; % 转换为行向量输出
for i=1:1:m
    sys(i)=S(i); % 输出每个权值变化率
end

function sys=mdlOutputs(t,x,u)
global c b m kv kp

qd=u(1); % 期望位置u(1)
dqd=cos(t); % 期望速度
ddqd=-sin(t); % 期望加速度

q=u(2); % 实际位置u(2)
dq=u(3); % 实际速度u(3)

e=q-qd; % 位置误差
de=dq-dqd; % 速度误差

M=10; % 名义模型惯量

tol1=M*(ddqd-kv*de-kp*e); % 基于误差反馈的控制力矩 tol1第一部分

xi=[e;de]; % RBF网络输入
h=zeros(m,1); % RBF输出初始化，创建一个m×1的零向量
for j=1:1:m
    h(j)=exp(-norm(xi-c(:,j))^2/(2*b^2)); % 计算每个隐藏节点输出
end

d=-15*dq-30*sign(dq); % 非线性摩擦扰动模型 disturbance term
f=d; % 实际未知扰动

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S=3; % 控制器选择：1CTC名义控制，2精确补偿，3 RBF补偿
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if S==1               % Nominal model based controller（无扰动补偿）
    fn=0; % 不估计扰动
    tol=tol1; % 只使用反馈控制
elseif S==2           % Modified computed torque controller（已知扰动补偿）
    fn=0;
    tol2=-f; % 直接提前抵消已知扰动
    tol=tol1+tol2; % 加入补偿项
elseif S==3           % RBF compensated controller（未知扰动RBF补偿）
    if m==5
    th=[x(1) x(2) x(3) x(4) x(5)]'; % 读取RBF权值
    elseif m==19
    th=[x(1) x(2) x(3) x(4) x(5) x(6) x(7) x(8) x(9) x(10) x(11) x(12) x(13) x(14) x(15) x(16) x(17) x(18) x(19)]'; % 读取19个权值
end
    fn=th'*h; % RBF网络估计未知扰动
    tol2=-fn; % 补偿估计扰动
    tol=tol1+1*tol2; % 总控制输入
end

sys(1)=tol; % 输出控制力矩
sys(2)=f; % 输出真实扰动
sys(3)=fn; % 输出RBF估计扰动