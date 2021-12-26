% load images
num_batches=5;
num_samples=num_batches*10000;
num_pix=3072;
data_=zeros(num_samples,num_pix);
labels_=zeros(num_samples,1);
for i=1:num_batches
    fname=['./cifar-10-batches-mat/data_batch_' num2str(i) '.mat'];
    load(fname, 'data', 'labels');
    
    proc_data=zeros(10000,32,32,3);
    for j=1:10000
        img_raw=data(j,:);
        ch_r=reshape(img_raw(1:1024),32,32);
        ch_g=reshape(img_raw(1025:2048),32,32);
        ch_b=reshape(img_raw(2049:3072),32,32);
        img=cat(3,ch_r,ch_g,ch_b);
        img=imrotate(img,-90,'bilinear');
        
        imshow(img);
        truesize([500 500]);
        w=waitforbuttonpress;
        
        proc_data(j,:,:,:)=img;
        imshow(squeeze(proc_data(j,:,:,:)));
        truesize([500 500]);
        w=waitforbuttonpress;
    end
    
    i_start=(i-1)*10000+1;
    i_end=i*10000;
    data_(i_start:i_end,:)=data;
    labels_(i_start:i_end)=labels;
end

% preprocess
f_data=zeros(num_samples,32,32,3);
for i=1:num_samples
    
end

% load training data
[trainInd,valInd,testInd]=dividerand(num_samples,0.6,0.2,0.2);
XTrain=data_(trainInd(:),:);
YTrain=labels_(trainInd(:));

% define nn config
layers=[
    featureInputLayer(num_feat,'Name','input')
    
    fullyConnectedLayer(num_lay,'Name','fc1')
    reluLayer('Name','relu1')
    
    fullyConnectedLayer(num_lay,'Name','fc2')
    reluLayer('Name','relu2')
    
    fullyConnectedLayer(num_label,'Name','fc3')
    softmaxLayer('Name','sm')
    
    classificationLayer('Name','classification')
];

options=trainingOptions('sgdm', ...
    'MaxEpochs',100, ...
    'InitialLearnRate',0.04, ...
    'MiniBatchSize',100, ...
    'Plots','training-progress' ...
);

% traing nn