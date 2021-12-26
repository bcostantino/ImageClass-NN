function [XTrain,YTrain,XVal,YVal,XTest,YTest]=prepNetworkData(trainPerc,valPerc)
    fname='./procData.mat';
    if isfile(fname)
        load(fname,'procData','labels','numSamples');
    else
        [procData,labels,numSamples]=loadData(5);
    end
    
    [trainInd,valInd,testInd]=dividerand(numSamples,trainPerc,valPerc,(1-trainPerc-valPerc));
    
    % prep training data
    XTrain=permute(procData(trainInd(:),:,:,:),[2 3 4 1]);
    YTrain=categorical(labels(trainInd(:)));
    
    % prep validation data
    XVal=permute(procData(valInd(:),:,:,:),[2 3 4 1]);
    YVal=categorical(labels(valInd(:)));
    
    % prep test data
    XTest=permute(procData(testInd(:),:,:,:),[2 3 4 1]);
    YTest=categorical(labels(testInd(:)));
    
    % save data
    save('./networkData.mat','XTrain','YTrain','XVal','YVal','XTest','YTest');
end