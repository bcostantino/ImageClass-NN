% main.m
%  main MATLAB script to handle training the NN

fname='/networkData';
if isfile(fname)
    load('./networkData.mat','XTrain','YTrain','XVal','YVal','XTest','YTest');
else
    trainPerc=0.6;
    valPerc=0.0;
    [XTrain,YTrain,XVal,YVal,XTest,YTest]=prepNetworkData(trainPerc,valPerc);
end

for i=1:1000
    imshow(XTrain(:,:,:,i));
    truesize([200 200])
    title(YTrain(i));
    pause(5);
end