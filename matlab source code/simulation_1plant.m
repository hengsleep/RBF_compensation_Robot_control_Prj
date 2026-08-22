function [sys,x0,str,ts]=s_function(t,x,u,flag)
switch flag,
case 0,
    [sys,x0,str,ts]=mdlInitializeSizes; % 初始化S-function参数
case 1,
    sys=mdlDerivatives(t,x,u); % 计算系统状态导数
case 3,
    sys=mdlOutputs(t,x,u); % 输出系统状态
case {2, 4, 9 }
    sys = []; % 未使用的flag类型
otherwise
    error(['Unhandled flag = ',num2str(flag)]);
end

function [sys,x0,str,ts]=mdlInitializeSizes

sizes = simsizes;
sizes.NumContStates  = 2; % 连续状态数量：q和dq
sizes.NumDiscStates  = 0; % 无离散状态
sizes.NumOutputs     = 2; % 输出：[位置q,速度dq]
sizes.NumInputs      = 3; % 输入：[控制力矩tol,其他输入占位]
sizes.DirFeedthrough = 0; % 输出不直接依赖输入，只由状态决定
sizes.NumSampleTimes = 0; % 连续系统

sys=simsizes(sizes);

x0=[0.6;0]; % 初始状态：[初始位置q,初始速度dq]

str=[];
ts=[];

function sys=mdlDerivatives(t,x,u)

M=10; % 系统名义惯量参数

d=-15*x(2)-30*sign(x(2)); % 非线性摩擦扰动模型

tol=u(1); % 输入控制力矩

% 状态方程：
% x1=q
% x2=dq
% x1_dot=dq
% x2_dot=(tol+d)/M
sys(1)=x(2); % 位置导数，即速度
sys(2)=1/M*(tol+d); % 加速度，由控制力矩和扰动共同决定

function sys=mdlOutputs(t,x,u)

sys(1)=x(1); % 输出位置q，这里的sys和上面的mdlDerivatives中的sys不是一个，只是都代表公式的输出
sys(2)=x(2); % 输出速度dq