%% 新的去混响算法原型
% 针对2021/10/15前后，去混响暴露出的问题，重新编写去混响算法框架
% 暴露问题：
%         （1）算法过于冗余，有大量不必要的部分
%         （2）与外部模块存在耦合，容易出现问题
%         （3）参数的效果不明显
%         （4）某些设备在低频处会出现过长的混响拖尾，需要特殊对待
% 改动思路：
%         （1）curve曲线没有理论支持，且不方便进行分析，准备将其替换为混响衰减模型
%         （2）相干性继续保留，看看能不能结合混响扩散模型以及噪声场等进行完善
%         （3）加入波束，虽然我们不知道声源的具体方向，但是利用混响的扩散场特性，
%          通过多个方向的波束结果加权，一定程度上能够起到作用
%         （4）争对混响极其突出的子频带，使用RLS、LSL等自适应滤波器进行精细处理
% 20220308：整理代码，准备后续进行研究与改进

clc; clear all; %close all;

%% 宏变量
SAFE_DENOMINATOR = 1e-8; % 安全系数
SMOOTH_DEFAULT = 0.68; % 平滑系数
S_SPEED = 343; % 声速
D_MIC = 0.035; % 麦间距（线阵）或者半径（圆阵）
LOWFREEQ = 0; % 是否对低频处进行特别处理
MIC_TYPE = 0; % 0:线阵 1:圆阵
TIME_MODEL = 1; % 是否启用基于时域上

%% 输入输出
inputFileName = "data/audio_2channel.wav";
outputFileName = "res.wav";

[inputSig, fs] = audioread(inputFileName);
inputSig = inputSig(1:end, :);
[nLen, nChan] = size(inputSig);

if 5 == nChan || 6 == nChan
    inputSig = inputSig(1: end, [1,2,3,4]); % 舍弃回采通道
    nChan = 4;
    IS_SINGLE = 0;
elseif 1 == nChan
    IS_SINGLE = 1; % 单通道情况下，波束和相干性的算法就不能使用了
end
    
outputSig = zeros(nLen, 1);


%% stft参数设置
wTime = 32; % 单位:ms
wLen = floor(wTime * fs / 1000);
win = sqrt(hanning(wLen));
overlap = 0.5;
len1 = floor(wLen * overlap);
len2 = wLen - len1;
nFrame = floor((nLen - wLen)/len2);
nFFT = wLen;
nFreq = floor(wLen/2 + 1);
frequency     = linspace(0, fs/2, nFreq)'; % 每个频率帧对应的频率大小
Gres = zeros(nFreq, 1); % 最后增益
linecoeff = linspace(0.5, 1.2, nFreq);

%% 背景噪声估计
% 最小能量追踪估计噪声
% 参数
alpha   = 0.7;
beta    = 0.96;
gamma   = 0.998;
alpha2  = (1 - gamma) / (1 - beta);
% 分配空间
pnk     = zeros(nFreq, 1);
pxk_old = zeros(nFreq, 1);
pnk_old = zeros(nFreq, 1);
pxk     = zeros(nFreq, 1);

%% 指数衰减模型
T60 = 1;
D = 3;
coeff = 2;
sigma_x2      = zeros(nFreq, 1);
sigma_r2_pre  = zeros(nFreq, 1);
sigma_x2_buff = zeros(nFreq, D);     
Intermediate  = -6 * log(10) * (nFreq - 1) / (T60 * fs) * ones(nFreq, 1);


%% 阵列设置
theta = [0 90 180 270] / 180 * pi; % 入射角
n_angle = length(theta);

% 麦克风间距
micPos = zeros(nChan, 2); % 麦克风二维坐标
arraySpacing = zeros(nChan, nChan);
if 0 == MIC_TYPE % 线阵
    for iChan = 1 : nChan
        %micPos(iChan, 1) = D_MIC * (iChan - 1);
        micPos(iChan, 1) = D_MIC * (nChan - 2 * iChan + 1);
        micPos(iChan, 2) = 0;
    end
elseif 1 == MIC_TYPE % 圆阵
    for iChan = 1 : nChan
        sv_psi = 2 * pi / nChan * (iChan - 1);
        micPos(iChan, 1) = D_MIC * cos(sv_psi);
        micPos(iChan, 2) = D_MIC * sin(sv_psi);
    end
    
end

for iChan1 = 1 : nChan
    for iChan2 = iChan1 + 1 : nChan
        arraySpacing(iChan1, iChan2) = norm(micPos(iChan1, :)-micPos(iChan2,:), 2);
        arraySpacing(iChan2, iChan1) = arraySpacing(iChan1, iChan2);
    end
end

% ------------------------ Blocking-PSD-LS --------------------------
% 入射信号
steerMat = zeros(nChan, n_angle, nFreq); % 方向矢量矩阵
diffuseMat = zeros(nChan, nChan, nFreq); % 扩散场矩阵
BMat = zeros(nChan, nChan-1, n_angle, nFreq); % 屏蔽矩阵
diffuseBlockMat = zeros(nChan - 1, nChan - 1, n_angle, nFreq); % 阻塞扩散场矩阵

for iFreq = 1 : nFreq
    diffuse = sinc(2 * frequency(iFreq) * arraySpacing / S_SPEED); % 扩散场噪声
    diffuseMat(:,:,iFreq) = diffuse;
    for i_angle = 1 : n_angle
        if 0 == MIC_TYPE
            sv_tau  = micPos(:,1)*cos(theta(i_angle)) / S_SPEED;
        elseif 1 == MIC_TYPE
            for iChan = 1 : nChan
                sv_psi = 2 * pi / nChan * (iChan - 1);
                sv_tau = D_MIC * cos(theta(i_angle) - sv_psi) / S_SPEED;
            end
        end
        steerVec = exp(1i * 2 * pi * frequency(iFreq) * sv_tau); % 方向矢量
        steerMat(:, i_angle, iFreq) = steerVec;
        
        B = eye(nChan) - steerVec / (steerVec' * steerVec) * steerVec';
        B = B(:, 1:end-1);
        BMat(:, :, i_angle, iFreq) = B;
        
        diffuseBlock = real(B' * diffuse * B);
        diffuseBlockMat(:, :, i_angle, iFreq) = diffuseBlock;
    end
end

psd_U = zeros(nChan-1, nChan-1, n_angle, nFreq);
psd_Y = zeros(nChan, nChan, n_angle, nFreq);
sigma_r1 = zeros(nFreq, 1);

% ---------------------------- 相干性 -------------------------------
Pxx_buff = zeros(nChan, nChan, iFreq);
nPairs = nChan * (nChan-1) / 2;
Ctemp = zeros(nPairs, iFreq);
Cxxs = zeros(nPairs, iFreq);
Cohs = zeros(nFreq, 1);
CDRs = zeros(nPairs, iFreq);

%% 自适应滤波器
Delay = 3;
Lg = 10;
forgetting = 0.98;
fLimit = floor(1000/(0.5 * fs) * nFreq); % 针对低频进行处理

% ---------------------------- WRLS ----------------------------
x_buffer            = zeros(nFreq, nChan*(Lg+Delay-1)); % 观测信号的缓冲区
WRLS_Pred_Filter    = zeros(Lg*nChan, nFreq, nChan);     % 为滤波器系数分配空间
WRLS_Pred_Psi       = zeros(nChan, nChan, Lg, nFreq);    % 为WRLS的Psi分配空间\
e                   = zeros(nFreq, nChan);      % 残差
sigma_r2            = zeros(nFreq, 1);          % 信号混响部分的psd
devOutSpec          = zeros(nFreq, nChan);
Gwrls               = ones(nFreq, 1);

for iFreq =  1:nFreq
    for iL = 1:Lg
        WRLS_Pred_Psi(:,:,iL,iFreq) = eye(nChan);  % WRLS的Psi初始化
    end
end


%% 开始处理
nStart = 1;
for iFrame = 1 : nFrame
    %% stft
    yOneFrame = inputSig(nStart : nStart + wLen - 1, :);
    ySpec = fft(yOneFrame.*win, wLen);
    ySpec(nFreq + 1 : end, :) = [];
    x2 = sum(abs(ySpec).^2,2)/nChan + 1e-6;
    
    %% 噪声估计
    % --------------------- 最小能量追踪估计噪声psd ---------------------
    pxk = alpha * pxk_old + (1 - alpha) * x2;
    for iFreq = 1 : nFreq
        if pnk_old(iFreq) <= pxk(iFreq)
            pnk(iFreq) = (gamma.*pnk_old(iFreq)) + (alpha2.*(pxk(iFreq) - beta.*pxk_old(iFreq)));
        else
            pnk(iFreq) = pxk(iFreq);
        end
    end
    pxk_old = pxk;
    pnk_old = pnk;
    noise_ps = pnk;
    
    %% 自适应滤波
    if LOWFREEQ
        % 观测信号的缓冲区 和 预测缓冲区 进行更新（包含全部频率子带）
        wVec = x2;
        xtaun_idx               = (Delay - 1) * nChan + 1;
        xtaun                   = x_buffer(:, xtaun_idx:end);
        x_buffer                = [ySpec x_buffer(:, 1:end-nChan)];
        devOutSpec              = ySpec;
        
        for iFreq = 1 : fLimit
            WRLS_Pred_Vec_temp = xtaun(iFreq, :).';  % 获取当前滤波器系数对应的缓冲区
            rcounter = zeros(nChan, 1);
            for iL = 1: Lg
                Psi_t             = reshape(WRLS_Pred_Psi(:, :, iL, iFreq), nChan, nChan); % 获取当前的Psi矩阵
                Psi_t             = 0.5 * (Psi_t + Psi_t'); 
                WRLS_Pred_Vec     = WRLS_Pred_Vec_temp((iL-1)*nChan+1: iL*nChan);
                nominator         = Psi_t * WRLS_Pred_Vec; % 4M^2
                donominator       = real(WRLS_Pred_Vec' * nominator) + forgetting * wVec(iFreq); % 4M^2 + 2M
                K                 = nominator / donominator; % 4M                                      
                WRLS_Pred_Psi(:,:, iL, iFreq) = (Psi_t - K * WRLS_Pred_Vec' * Psi_t) / forgetting; % 
                % 对每一通道进行计算
                for k = 1:nChan
                    % kalman更新
                    filter_t             = WRLS_Pred_Filter((iL-1)*nChan+1: iL*nChan, iFreq, k);        % 获取当前滤波器系数
                    filter_delta         = K * (devOutSpec(iFreq,k)' - WRLS_Pred_Vec'*filter_t);    % 前后帧系数变化（残差）
                    filter_t             = filter_t + filter_delta;                                     % 更新滤波器系数
                    r                    = filter_t' * WRLS_Pred_Vec;   
                    devOutSpec(iFreq, k) = devOutSpec(iFreq,k) - r;
                    rcounter(k)          = rcounter(k) + r;
                    WRLS_Pred_Filter((iL-1)*nChan+1: iL*nChan, iFreq, k) = filter_t;
                end
            end
            sigma_r2(iFreq) = sum(abs(rcounter).^2) / nChan;
        end
    else
        sigma_r2 = zeros(nFreq, 1);
    end
    
    
    
    
    
    
    %% 矩阵相干性
    ySpec_T = ySpec.';
    for iFreq = 1 : nFreq
        Pxx = Pxx_buff(:, :, iFreq);
        Pxx = SMOOTH_DEFAULT * Pxx + (1 - SMOOTH_DEFAULT) * (ySpec_T(:, iFreq) * ySpec_T(:, iFreq)');
        
        % ==============================================
        % TODO：尝试利用背景噪声信息，不过目前来看，效果不佳
%         Pxx = Pxx - noise_ps(iFreq) *eye(nChan);
%         for iChan = 1 : nChan
%             Pxx(iChan, iChan) = max(Pxx(iChan, iChan), 0.1 * x2(iFreq));
%         end
        % ==============================================      

        iPairs = 1;
        for iChan1 = 1 : nChan-1
           for iChan2 = iChan1 + 1: nChan
               Cxxs(iPairs, iFreq) = Pxx(iChan1, iChan2) / sqrt(real(Pxx(iChan1, iChan1)) * real(Pxx(iChan2, iChan2)));
               iPairs = iPairs + 1;
           end
        end
        Pxx_buff(:, :, iFreq) = Pxx;
    
        % no doa
        iPairs = 1;
        for iChan1 = 1 : nChan - 1
            for iChan2 = iChan1 + 1 : nChan
                Cnn = diffuseMat(iChan1,iChan2,iFreq);
                Cxx = Cxxs(iPairs, iFreq);
                if abs(Cxx)>1-1e-10
                    Cxx = (1-1e-10) * Cxx / abs(Cxx);
                end
                
                CDR_temp =  (-(abs(Cxx).^2 + Cnn.^2.*real(Cxx).^2 - Cnn.^2.*abs(Cxx).^2 - 2.*Cnn.*real(Cxx) + Cnn.^2).^(1/2) - abs(Cxx).^2 + Cnn.*real(Cxx))./(abs(Cxx).^2-1);
                CDR_temp = max(real(CDR_temp), 0);
                CDRs(iPairs, iFreq) = CDR_temp;
                iPairs = iPairs + 1;
            end
        end
        CDR = sum(CDRs)/nPairs;
    end
    
    
    
    %% 波束形成
    % --------------------- PSD-blocking ---------------------
    for iFreq = 1 : nFreq
        sum_d = 0;
        y = ySpec(iFreq, :).';
        d_buff = zeros(n_angle, 1);
        for i_angle = 1 : n_angle
            B = BMat(:,:,  i_angle, iFreq);
            u = B'*y;
            psd_u = psd_U(:, :, i_angle, iFreq);
            psd_u = beta * psd_u + (1 - beta) * (u * u');
            
            diffuseBlock = diffuseBlockMat(:, :, i_angle, iFreq);
            d2 = real(trace(diffuseBlock' * psd_u) / (trace(diffuseBlock' * diffuseBlock))+1e-8);
            sum_d = sum_d + d2;
            d_buff(i_angle) = d2;
            psd_U(:, :,i_angle, iFreq) = psd_u;
        end
        sigma_r1(iFreq) = sum_d / n_angle;
        %sigma_r1(iFreq) = min(d_buff);
    end
    
    %% 计算结果
    Gcdr = max(1 - (1./(CDR.' + 1)).^1, 0).^2;
    Gcdr = max(Gcdr,0);
    Gcdr = max(sqrt(Gcdr),0.1);
    
    Gbeam = n_angle * min(max(1 - sqrt(sigma_r1./x2), 0.02), 1);
    Gbeam = min(Gbeam, 1);
    Gbeam   = (Gbeam.^(1 - (linecoeff.'.*abs(Cxx)).^2));
    
    
    Gwrls(1:fLimit) = min(max(1 - sqrt(sigma_r2(1:fLimit)./x2(1:fLimit)), 0.01), 1);
    
    %res = ySpec(:,1).*Gres;
    Gtemp = Gbeam.*Gwrls;
    Gtemp = smooth(Gtemp, 3);
    
    Gres = SMOOTH_DEFAULT * Gres + (1 - SMOOTH_DEFAULT) * Gtemp;
    
    res = ySpec(:,1).*Gres;
    outputSig(nStart:nStart+wLen-1)=outputSig(nStart:nStart+wLen-1)+  win.*real(ifft( [res;conj(res(end-1:-1:2))]));
    nStart = nStart+len2;
end
    
%% 语谱图可视化    
myfig(inputSig(:,1), fs);
myfig(outputSig(:,1), fs);
audiowrite(outputFileName, outputSig, fs);
    
