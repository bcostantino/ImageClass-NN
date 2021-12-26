% prepNetworkData Summary of this function goes here
%   Detailed explanation goes here
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

% loadData Summary of this function goes here
%   Detailed explanation goes here
function [procData,labels,numSamples] = loadData(numBatches)
    numSamples=numBatches*10000;
    procData=uint8(zeros(numSamples,32,32,3));
    labels_=zeros(numSamples,1);
    for i=1:numBatches
        fname=['./cifar-10-batches-mat/data_batch_' num2str(i) '.mat'];
        load(fname, 'data', 'labels');

        for j=1:10000

            % preprocess each image
            img_raw=data(j,:);
            ch_r=reshape(img_raw(1:1024),32,32);
            ch_g=reshape(img_raw(1025:2048),32,32);
            ch_b=reshape(img_raw(2049:3072),32,32);
            img=cat(3,ch_r,ch_g,ch_b);
            img=imrotate(img,-90,'bilinear');

            ind=(i-1)*1000+j;
            procData(ind,:,:,:)=img;
        end

        i_start=(i-1)*10000+1;
        i_end=i*10000;
        labels_(i_start:i_end)=labels;
    end
    labels=labels_;
    save('./procData.mat','procData','labels','numBatches','numSamples');
end