%% 基于指数衰减模型的的去混响算法
% ref:[1]Braun S, Kuklasiński A, Schwartz O, et al. 
%        Evaluation and comparison of late reverberation power spectral density estimators[J]. 
%        IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2018, 26(6): 1056-1071.
% 在原算法基础上，做了一些微小的改动，以减少过消除的情况
clc; clear; close all;

%% 宏变量
SAFE_DENOMINATOR = 1e-8;
D_MIC = 0.035;
S_SPEED = 343;
MIC_TYPE = 0; % 0:线阵  1:圆阵

%% 输入输出
inputFileName = "data/audio_2channel.wav"; % 注意输入数据不要包含回采通道
outputFileName = "res2.wav";

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
D = 3;
T60 = 2;
T60db = ones(nFreq, 1) * T60;                             % 混响时间
Intermediate = -6 * log(10) * (nFreq-1) ./ (T60db * fs);  % 中间变量
sigma_x2_buff = zeros(nFreq, D);                        % 追踪晚期混响PSD的变量
sigma_r2_pre = zeros(nFreq, 1);
ED_beta = 0.5;      % 追踪晚期混响PSD的参数
SS_beta = 0.3;      % 追踪早期混响PSD的参数

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
Gres         = ones(nFreq, 1);
Gpre          = ones(nFreq, 1);
pureSig       = zeros(nLen, 1);
Cnn           = zeros(nFreq, 1);
pnk      = zeros(nFreq, 1);
% 扩散场噪声相干性
rtmp = 2 * pi * d_mic * fs / (2 * c * nFreq);
Cnn(1) = 1;
for iFreq = 2 : nFreq  
    Cnn(iFreq) = abs(sin((iFreq - 1) * rtmp) / ((iFreq - 1) * rtmp));
end

a_dd    = 0.98;     % 先验SNR平滑系数
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
    
    % 利用指数衰减模型来估算混响信号psd [1]-V-A.Statistical Temporal Model
    if (iFrame - D <=0)
        sigma_x_Ne2 = x2;
    else
        sigma_x_Ne2 = sigma_x2_buff(:,end);
    end
    if iFrame==1
        sigma_x2_pre = x2;
    else
        sigma_x2_pre = sigma_x2_buff(:,1);
    end
    % 信号psd估算
    sigma_x2 = ED_beta * sigma_x2_pre + (1-ED_beta) * x2;
    % 信号后期混响psd估算
    sigma_r2 = exp(Intermediate) .* sigma_r2_pre + exp(Intermediate*D) .* sigma_x_Ne2;  % [1]-(55)
    % 信号所有混响psd估算
    % 这里就是与论文方法不一样的地方
    sigma_r2_all = exp(Intermediate) .* sigma_r2_pre + exp(Intermediate) .* sigma_x2_pre;
    sigma_r2_pre = sigma_r2;
    
    % 计算去除后期混响增益
    posteri = x2./ (sigma_r2 + SAFE_DENOMINATOR);
    posteri_prime = posteri - 1;
    posteri_prime(posteri_prime < 0)= 0;
    priori = a_dd * (GtempPrev.^ 2).* posteri_prev + ...
            (1-a_dd)* posteri_prime;
    Gnew = priori./ (1+ priori); % gain function   
    GtempPrev= Gnew; 
    posteri_prev = posteri;
    
    % 计算去除所有混响增益
    posteri = x2./ (sigma_r2_all + SAFE_DENOMINATOR);
    posteri_prime = posteri - 1;
    posteri_prime(posteri_prime < 0)= 0;
    priori = a_dd * (Gtemp2Prev.^ 2).* posteri2_prev + ...
            (1-a_dd)* posteri_prime;
    Gpure = priori./ (1+ priori); % gain function   
    Gtemp2Prev= Gpure; 
    posteri2_prev = posteri;
    
    sigma_x2_buff = [x2.*(Gpure.^2) sigma_x2_buff(:,1:end-1)];
   
    % 对信号进行校准，避免过消除
    Gnew(Gnew < 0.025) = 0.025;
    calibration = (1 - Coh.^coCoeff) .* coef;
    Gnew  = Gnew.^calibration;     
    
    Gres = (1 - smoothCoeff1) * Gres + smoothCoeff1 * Gnew;
    Gres(Gres > 1.0) = 1.0;
    Gres(Gres < 0.025) = 0.025;
    
    res = Gres .* ySpec(:, 1);
    res(257) = imag(res(1));
    res(1) = real(res(1));
    outputSig(nStart:nStart+wLen-1,1) = outputSig(nStart:nStart+wLen-1,1)...
                                         + win.*real(ifft( [res;conj(res(end-1:-1:2))]))*nFFT/32767;
    
    nStart = nStart+nHop;
end
toc;

%% 语谱图可视化    
%myfig(inputSig(:,1)/32767, fs);
%myfig(outputSig(:,1), fs);
audiowrite(outputFileName, outputSig, fs);