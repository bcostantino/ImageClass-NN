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
    'Plots','training-progress' ...
);

if isfile('./training_data.mat')
    load('./training_data.mat','X','Y');
else
    [X,Y]=get_training_data('gray');
end

XTrain=X;
YTrain=categorical(Y);

if isfile('./trained_net.mat')
    load('./trained_net.mat','net');
else
    net=trainNetwork(X,YTrain,layers,options);
    save('./trained_net.mat','net');
end

analyzeNetwork(net);
%end