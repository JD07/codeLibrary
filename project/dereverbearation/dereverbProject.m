%% 该代码用于与工程项目方法保持一致，便于问题分析以及算法改进
% 20220308：该代码建立，与工程代码算法保持尽可能一致，目前使用的是curve算法
clc; clear; close all;

%% 宏变量
SAFE_DENOMINATOR = 1e-8;
D_MIC = 0.035;
S_SPEED = 343;
MIC_TYPE = 0; % 0:线阵  1:圆阵

%% 输入输出
inputFileName = "data/audio_2channel.wav"; % 注意输入数据不要包含回采通道
outputFileName = "res.wav";

[inputSig, fs] = audioread(inputFileName, "Native");
inputSig = double(inputSig);
[nLen, nChan] = size(inputSig);
if 1 == nChan
   error("at least 2 channels"); 
end

%% stft参数设置
wLen          = 512;                      % 窗函数长度
win           = sin_win(wLen);                    % 窗函数
nHop          = 256;                          % Step shift
nFrame        = floor((nLen-wLen)/nHop);
nFFT          = wLen;
nFreq         = wLen/2;
outputSig     = zeros(nLen+wLen, 1); % 输出信号

%% 最小能量追踪估计噪声
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

%% ======================= 混响指数衰减模型 ======================
lenc = 20;
iw = 1 : 1 : lenc;
T60 = 2; % 基础混响时间 
alpha3 = 3*log(10)./(T60*fs);  
w = exp(-2 * alpha3 * nFreq.*iw);
a_dd = 0.98;     % 先验SNR平滑系数
gap = 2;
delay = 2;


%% 相干性设置
% 参数
coCoeff          = 0.5;                % 互相干性校准程度，小于1时会降低去混响程度，大于1会提高去混响程度
smoothCoeff1     = 0.68;               % 平滑系数
smoothCoeff2     = 0.6;               % 平滑系数
d_mic            = D_MIC;
c                = S_SPEED;                % speed of sound [m/s]
bias             = 1.2;
coef = 1.1 - 0.3 * [0:nFreq-1]'/nFreq;

% 参与计算互相干性的通道编号，需要手动调整
if 0 == MIC_TYPE
    nPairs = nChan - 1;
elseif 1 == MIC_TYPE
    nPairs = nChan;
end

% 分配空间
Pxx           = zeros(nFreq, nChan); 
Cxx           = zeros(nFreq, 1);
Ctemp         = zeros(nFreq, nPairs);
CDR           = zeros(nFreq, 1);
CxxBuffer     = zeros(nFreq, nPairs);
CohBuffer     = zeros(nFreq, nPairs);
G             = zeros(nFreq, 1);
Gnew          = zeros(nFreq, 1);
Gres          = ones(nFreq, 1);
Gres2         = ones(nFreq, 1);
Gpre          = ones(nFreq, 1);
Cnn           = zeros(nFreq, 1);
specBuffer    = zeros(nFreq, lenc);
pnk      = zeros(nFreq, 1);
% 扩散场噪声相干性
rtmp = 2 * pi * d_mic * fs / (2 * c * nFreq);
Cnn(1) = 1;
for iFreq = 2 : nFreq  
    Cnn(iFreq) = abs(sin((iFreq - 1) * rtmp) / ((iFreq - 1) * rtmp));
end

GtempPrev = zeros(nFreq, 1);
posteri_prev = zeros(nFreq, 1);
Gtemp2Prev = zeros(nFreq, 1);
posteri2_prev = zeros(nFreq, 1);

%% ======================= Start processing =======================
tic;
index = 1; % 缓存区指针位置
nStart = 1;
for iFrame = 1:nFrame
    % --------------------- 对当前语音片段进行stft ---------------------
    yOneFrame = inputSig(nStart:nStart+wLen-1,:);
    ySpec = fft(yOneFrame.*win, wLen)./ nFFT;
    ySpec(1,:) = ySpec(1,:) + 1i* ySpec(nFreq+1,:);
    ySpec(nFreq+1:end, :) = [];
    x2 = mean(abs(ySpec).^2, 2);
    Gamma_n = pnk.*Cnn;
        
    % ------------------------ 通道互相干计算 -----------------------
    Pxx = smoothCoeff1 * Pxx + (1 - smoothCoeff1) * abs(ySpec).^2;
    % 对角线计算
    X1 = ySpec(:, 1);
    pxx1 = Pxx(:, 1);
    XA = X1;
    pxxA = pxx1;
    Coh = zeros(nFreq, 1);
    for iChan = 2 : nChan
        XB = ySpec(:, iChan);
        pxxB = Pxx(:, iChan);
        Ctemp(:, iChan-1) = smoothCoeff2 * Ctemp(:, iChan-1) + (1 - smoothCoeff2) * XA.*conj(XB);
        Coh = Coh + max(abs(Ctemp(:, iChan-1) - Gamma_n),0) ./ sqrt(pxxA.*pxxB);
        XA = XB;
        pxxA = pxxB;
    end
    % 圆阵额外需要计算通道n-通道1的相关性
    if nPairs == nChan
        Ctemp(:, nChan) = smoothCoeff2 * Ctemp(:, nChan) + (1 - smoothCoeff2) * XA.*conj(X1);
        Coh = Coh + max(abs(Ctemp(:, nChan) - Gamma_n),0) ./ sqrt(pxxA.*pxx1);
    end
    Coh = Coh/nPairs;
    Coh = Coh/bias;
    Coh = min(Coh, 1);
    
    % --------------------- 最小能量追踪估计噪声psd ---------------------
    % 理论上噪声追踪应该放在相干性前面，但是在实际工程代码的优化中，为了减少数组的读取
    % 将这部分计算移动到了后面
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
    
    % --------------- 利用瑞利方差系数与缓存计算混响部分psd ---------
    specReverb_1 = zeros(nFreq, 1);
    specReverb_D = zeros(nFreq, 1);
    tempIndex = index-1;
    if tempIndex <= 0
       tempIndex = tempIndex + lenc;
    end
    
    for k = 1 : gap : lenc
        specReverb_1 = specReverb_1 + w(k) .* specBuffer(:, tempIndex);
        if (k > delay) 
            specReverb_D = specReverb_D + w(k) .* specBuffer(:, tempIndex);
        end
        tempIndex = tempIndex - gap;
        if tempIndex <= 0
            tempIndex = tempIndex + lenc;
        end
    end
    
    specReverb_1 = min(specReverb_1, 0.5 * x2);
    specReverb_D = min(specReverb_D, 0.8 * x2);
    
    
    % 计算所有混响部分
    posteri = x2./ (specReverb_1 + SAFE_DENOMINATOR);
    posteri_prime = posteri - 1;
    posteri_prime(posteri_prime < 0)= 0;
    priori = a_dd * (GtempPrev.^ 2).* posteri_prev + ...
            (1-a_dd)* posteri_prime;
    % 计算辅助信号
    Gpure = priori./ (1+ priori); % gain function   
    GtempPrev= Gpure; 
    posteri_prev = posteri;
    
    
    % 计算晚期混响部分
    posteri = x2./ (specReverb_D + SAFE_DENOMINATOR);
    posteri_prime = posteri - 1;
    posteri_prime(posteri_prime < 0)= 0;
    priori = a_dd * (Gtemp2Prev.^ 2).* posteri2_prev + ...
            (1-a_dd)* posteri_prime;
    % 计算辅助信号
    Gnew = priori./ (1+ priori); % gain function   
    Gtemp2Prev= Gnew; 
    posteri2_prev = posteri;
    
    Gnew(Gnew < 0.025) = 0.025;
    
    calibration = (1 - Coh.^coCoeff) .* coef;
    
    Gnew  = Gnew.^calibration;
    
    Gres = (1 - smoothCoeff1) * Gres + smoothCoeff1 * Gnew;
    Gres(Gres > 1.0) = 1.0;
    Gres(Gres < 0.025) = 0.025;
    
    % 工程代码里有问题，这里应该要乘Gpure的平方
    specBuffer(:,index) = x2 .* (Gpure.^2);
    index = index + 1;
    if (index > lenc)
        index = 1;
    end
    
    res1 = Gres .* ySpec(:, 1);
    res1(257) = imag(res1(1));
    res1(1) = real(res1(1));
    
    outputSig(nStart:nStart+wLen-1,1) = outputSig(nStart:nStart+wLen-1,1)...
                                         + win.*real(ifft( [res1;conj(res1(end-1:-1:2))]))*nFFT/32767;
    
    nStart = nStart+nHop;
end
toc;

%% 语谱图可视化    
%myfig(inputSig(:,1)/32767, fs);
%myfig(outputSig(:,1), fs);
audiowrite(outputFileName, outputSig, fs);