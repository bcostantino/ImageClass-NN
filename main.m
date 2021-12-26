% main.m
%  main MATLAB script to handle training the NN
clear;

fname='/networkData';
if isfile(fname)
    load('./networkData.mat','XTrain','YTrain','XVal','YVal','XTest','YTest');
else
    trainPerc=0.6;
    valPerc=0.2;
    [XTrain,YTrain,XVal,YVal,XTest,YTest]=prepNetworkData(trainPerc,valPerc);
end

