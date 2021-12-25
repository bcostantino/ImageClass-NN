load('./cifar-10-batches-mat/data_batch_1.mat');

img_num=1;

while 1>0
    % prepare image
    img_raw=data(img_num,:);
    ch_r=reshape(img_raw(1:1024),32,32);
    ch_g=reshape(img_raw(1025:2048),32,32);
    ch_b=reshape(img_raw(2049:3072),32,32);
    img=imrotate(cat(3,ch_r,ch_g,ch_b),-90,'bilinear');
    
    % plot image
    ax1=subplot(2,2,1);
    imshow(img);
    title(labels(img_num));
    
    % plot grayscale
    ax2=subplot(2,2,2);
    img_g=rgb2gray(img);
    imshow(img_g);
    title('Grayscale');
    
    % plot edge
    ax3=subplot(2,2,3);
    img_e=edge(img_g,'Canny');
    imshow(img_e);
    title('w/ Canny edge det');
    
    % plot edge 2
    ax4=subplot(2,2,4);
    img_e2=edge(img_g,'Prewitt');
    imshow(img_e2);
    title('w/ Prewitt edge det');
    
    % wait for key press and show new image
    w=waitforbuttonpress;
    img_num=img_num+1;
end