function [X,Y]=get_training_data()
    load('./cifar-10-batches-mat/data_batch_1.mat', 'data', 'labels');
    num_ex=size(data,1);
    X=zeros(num_ex,size(data,2)/3);
    Y=labels;
    for i=1:num_ex
        % get color image from dataset
        img_raw=data(i,:);
        ch_r=reshape(img_raw(1:1024),32,32);
        ch_g=reshape(img_raw(1025:2048),32,32);
        ch_b=reshape(img_raw(2049:3072),32,32);
        im=cat(3,ch_r,ch_g,ch_b);

        % preproccess image
        img=imrotate(im,-90,'bilinear');
        img_g=rgb2gray(img);

        % extract features for training
        X(i,:)=extract_features(img_g);
    end

    % save training data
    save('./training_data.mat', 'X', 'Y');
end