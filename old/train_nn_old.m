%function [net]=train_nn()
num_feat=1024;
num_label=10;
num_lay=20;

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
    'Verbose',false, ...
    'Plots','training-progress' ...
);

fname1='./training_data_old.mat';
if isfile(fname1)
    load(fname1,'X','Y');
else
    [X,Y]=get_training_data();
end

XTrain=X;
YTrain=categorical(Y);

fname2='./trained_net_old.mat';
if isfile(fname2)
    load(fname2,'net');
else
    net=trainNetwork(X,YTrain,layers,options);
    save(fname2,'net');
end

analyzeNetwork(net);
%end