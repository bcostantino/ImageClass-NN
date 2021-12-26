% full_process.m
%   example script to test full data handling, and training processes

clear;

% load images
num_batches=5;
num_samples=num_batches*10000;
num_pix=3072;
proc_data=uint8(zeros(num_samples,32,32,3));
labels_=zeros(num_samples,1);
for i=1:num_batches
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
        proc_data(ind,:,:,:)=img;
    end
    
    i_start=(i-1)*10000+1;
    i_end=i*10000;
    labels_(i_start:i_end)=labels;
end

% load training data
[trainInd,valInd,testInd]=dividerand(num_samples,0.6,0.2,0.2);

XTrain=permute(proc_data(trainInd(:),:,:,:),[2 3 4 1]);
YTrain=categorical(labels_(trainInd(:)));
XVal=permute(proc_data(valInd(:),:,:,:),[2 3 4 1]);
YVal=categorical(labels_(valInd(:)));

% define nn config
layers=[
    imageInputLayer([32 32 3])
    
    convolution2dLayer(10,20,'Name','convolution1')
    reluLayer('Name','relu1')
    
    maxPooling2dLayer(2,'Name','pool1')
    dropoutLayer('Name','drop')
    
    fullyConnectedLayer(10,'Name','fc')
    softmaxLayer('Name','sm')
    classificationLayer('Name','class')
];

options=trainingOptions('sgdm', ...
    'MaxEpochs',10, ...
    'ValidationData',{XVal,YVal}, ...
    'ValidationFrequency',30, ...
    'InitialLearnRate',0.04, ...
    'Verbose',false, ...
    'Plots','training-progress');

% traing nn
net=trainNetwork(XTrain,YTrain,layers,options);