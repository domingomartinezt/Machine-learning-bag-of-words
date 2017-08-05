%% Initialization
clear ; close all; clc

%% Setup of parameters
                           %% Setup of parameters
num_labels = 3;             % 10 labels, from 1 to 10
lambda = 0;                 % Regularization parameter
iterations = 1400;          % Number of iterations gradient descent
alpha = 0.2;                % Steep size for gradient
folds = 10;                 % Number of folds to use on cross validation
words = 200;                % Number of words to use on bagOfWords
                           %% Optional parameters (y/n)
test_steepst_descent='y';   % Test gradian steepest descent 
test_optimal_lambda='y';    % Test different lambda parameters (time consuming)

%% =========== Part 0: Download the CALTECH dataset=============
% Location of the compressed data set
url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz';
% Store the output in a temporary folder
outputFolder = fullfile(tempdir, 'caltech101'); % define output folder

if ~exist(outputFolder, 'dir') % download only once
    disp('Downloading 126MB Caltech101 data set...');
    untar(url, outputFolder);
end

%% =========== Part 1: Loading and Visualizing Data =============
rootFolder = fullfile(outputFolder, '101_ObjectCategories');

fprintf('\nPart 1: Loading and Visualizing Data ...')
tic
categories = {'airplanes', 'ferry', 'laptop'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
tbl = countEachLabel(imds)
minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)

% Find the first instance of an image for each category
airplanes = find(imds.Labels == 'airplanes', 1);
ferry = find(imds.Labels == 'ferry', 1);
laptop = find(imds.Labels == 'laptop', 1);

% figure

subplot(1,3,1);
imshow(readimage(imds,airplanes))
subplot(1,3,2);
imshow(readimage(imds,ferry))
subplot(1,3,3);
imshow(readimage(imds,laptop))

f = size(imds.Files, 2);
n = size(imds.Files, 1);
l = size(imds.Files, 1);
fprintf('\n(Done %f)',toc);


%% =========== Part 2: Extracting Features Bad of Words =============

fprintf('\n\nPart 2: Extracting Features Bad of Words ...')

bag = bagOfFeatures(imageSet(imds.Files), 'VocabularySize',words);
img = readimage(imds, 1);
featureVector = encode(bag, img);

% Plot the histogram of visual word occurrences
figure
bar(featureVector)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')

f = size(featureVector, 2);
n = size(imds.Files, 1);

fprintf('(Done %f)',toc);
fprintf('\nNumber of features to use: %f',f);
fprintf('\nNumber of samples: %f', n);

%% =========== Part 3: Encoding Images =============

fprintf('\n\nPart 3: Encoding Images to Train and Test ...')
tic
% build my new feature vector with values of histogram of each image
X = zeros(n,f);
y = zeros(n, 1);
for i = 1:length (imds.Files)
    img = readimage(imds, i);
    X(i,:) = encode(bag, img);
    if (imds.Labels(i) == 'airplanes')
        y(i)=1;
    elseif (imds.Labels(i) == 'ferry')
        y(i)=2;
    elseif (imds.Labels(i) == 'laptop')
        y(i)=3;
    end
end


fprintf('(Done %f)',toc);

%% =========== Part 4: Training Logistic Regression =============
fprintf('\n\nPart 4: Cross Validation Training Logistic Regression steepestGradientDescent...')
if test_steepst_descent == 'y'
    tic 
    indices = crossvalind('Kfold',n,folds);
    error_training = zeros(folds, 1);
    error_test = zeros(folds, 1);
    for i = 1:folds
        fprintf('\nFold: %f', i);
        test = (indices == i); train = ~test;
        X_train = X(train,:);
        y_train = y(train,:);
        X_test = X(test,:);
        y_test = y(test,:);
        
        % run gradient descent
        [all_w, j_h] = steepestGradientDescent(X_train, y_train, alpha, iterations, num_labels, lambda);
        fprintf('(Done %f)',toc);
     
        pred = predict(all_w, X_train);
        pred_test = predict(all_w, X_test);
        
        error_training(i) =  mean(double(pred == y_train)) * 100;
        error_test(i) = mean(double(pred_test == y_test)) * 100;
    end
    fprintf('(Done %f)',toc);
    % Plot the convergence graph
    plotGradient(j_h,'Steepest Gradient Descent');   
    
    fprintf('\nTraining Set Accuracy: %f', mean(double(error_training)));
    fprintf('\nTest Set Accuracy: %f\n', mean(double(error_test))); 
else
    fprintf('(Desactived)');
end


%% ================ Part 5: Tests Optimal Lambda ================

fprintf('\n\nPart 5: Optimazing parameters, testing differents lambda...')
if test_optimal_lambda == 'y'
    tic
    lambda_test = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2];
    plotBiasVsVariance(X_train, y_train, X_test, y_test, alpha, iterations, num_labels, lambda_test, 1);
    fprintf('\n(Done %f)',toc);
else
    fprintf('(Desactived)');
end