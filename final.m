
imgSet=imageSet('F:\abdelrahman\model\data','recursive');
imds = imageDatastore('F:\abdelrahman\model\data','IncludeSubfolders',true,'LabelSource','foldernames')
tbl = countEachLabel(imds)  
categories = tbl.Label;



testSet=imageSet('F:\abdelrahman\model\test','recursive');
imds2 = imageDatastore('F:\abdelrahman\model\test','IncludeSubfolders',true,'LabelSource','foldernames')              %#ok
tbl = countEachLabel(imds2)  
categories = tbl.Label;

tic
bag = bagOfFeatures(imgSet);
objectdata = double(encode(bag,imgSet));
toc
objectImageData = array2table(objectdata);
objectType = categorical(repelem({imgSet.Description}', [imgSet.Count], 1));
objectImageData.objectType = imds.Labels;
classificationLearner

testobjectData = double(encode(bag,testSet));
testobjectData = array2table(testobjectData,'VariableNames',trainedClassifier.RequiredVariables);

actualojectType = imds2.Labels;

predictedOutcome = trainedClassifier.predictFcn(testobjectData);
correctPredictions = (predictedOutcome == actualojectType);
validationAccuracy = sum(correctPredictions)/length(predictedOutcome) %#ok

%% Visualize how the classifier works
figure(1);
random_num = randi(length(imds2.Labels));

img = imds2.readimage(random_num);

imshow(img)
% Add code here to invoke the trained classifier
imagefeatures = double(encode(bag, img));
% Find two closest matches for each feature
[bestGuess, score] = predict(trainedClassifier.ClassificationSVM,imagefeatures);
% Display the string label for img
if bestGuess == imds2.Labels(random_num)
	titleColor = [0 0.8 0];
else
	titleColor = 'r';
end
title(sprintf('Best Guess: %s; Actual: %s',...
	char(bestGuess),imds2.Labels(random_num)),...
	'color',titleColor)
