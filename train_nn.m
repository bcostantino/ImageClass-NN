num_feat=1024;
num_label=10;

layers=[
    featureInputLayer(num_feat,'Name','input')
    fullyConnectedLayer(num_label,'Name','fc')
    softmaxLayer('Name','sm')
    classificationLayer('Name','classification')
];

options=trainingOptions('sgdm', ...
    'MaxEpochs',8, ...
    'InitialLearnRate',0.03 ...
);
if isfile('./training_data.mat')
    load('./training_data.mat','X','Y');
else
    [X,Y]=get_training_data();
end

net=trainNetwork(X,Y,layers,options);