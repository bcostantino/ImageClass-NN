function [feat_vect]=extract_features(img)
    img_edges=edge(img,'Canny');
    feat_vect=img_edges(:);
end