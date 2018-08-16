%% Load the data base ? (imageSet(Database name , recursive) returns a vecor of imageset hrough a recursive search)
faceDatabase = imageSet('FACEDATABASE','recursive');

%% Display montage (displaying image in smaller and compact form)
figure;
montage(faceDatabase(1).ImageLocation);
title('Image of first person')

%% split database into training and test sets
[training,test] = partition(faceDatabase,[0.8,0.2]);

%% Extract and display HOG features
person = 5;
[hogFeature,visualization] = ...
    extractHOGFeatures(read(training(person),1));
figure;
subplot(2,1,1);
imshow(read(training(person),1));
title('input Face');
subplot(2,1,2);plot(visualization);
title('HOG Feature');

%% Extract HOG feature for training set
trainingFeature = zeros(size(training,2)*training(1).Count,4680);
featureCount = 1;
for i = 1: size(training,2)
    for j = 1: training(i).Count
        trainingFeatures(featureCount,:) = extractHOGFeatures(read(training(i),1)); 
        trainingLabel{featureCount} = training(i).Description;
        featureCount = featureCount + 1;
    end
    personIndex{i} = training(i).Description;
end

%% Class Classifier using fitcecoc
faceClassifier = fitcecoc(trainingFeatures,trainingLabel);

%% Test Images from Test Set
%person = 1;
for person = 1: 5
queryImage = read(test(person),1);
queryFeatures = extractHOGFeatures(queryImage);
personLabel = predict(faceClassifier,queryFeatures);

booleanIndex = strcmp(personLabel,personIndex);
integerIndex = find(booleanIndex);
subplot(1,2,1);imshow(queryImage); title('Query Face');
subplot(1,2,2);imshow(read(training(integerIndex),1));title('Matched Face');
figure;
end

