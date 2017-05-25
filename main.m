function [] = main( )
%Main function that will call KNN , Centroid and Linear regression
%classifier

[fileName1,pathName1] = uigetfile('*.txt','Select the training data file');

[fileName2,pathName2] = uigetfile('*.txt','Select the test data file');



trainData = csvread(strcat(pathName1,fileName1),1,0);
classData = csvread(strcat(pathName1,fileName1),0,0,[0, 0, 0, size(trainData,2)-1]);
testData = csvread(strcat(pathName2,fileName2));

trainDataSVM = trainData;
testDataSVM = testData;


% call centroid, KNN and Linear Regression
centroid_clustering = CentroidClustering(trainData,testData,classData)
KNN = KNNClassifier(trainData',testData',classData)
Linear = LinearRegression(trainData,classData,testData)

%using libsvm for all 3 classifiers
model = svmtrain(classData', trainData', '-s 0 -t 1');
[predicted_label_KNN, accuracy_KNN, dec_values_KNN] = svmpredict(KNN', testDataSVM', model );
[predicted_label_lin, accuracy_lin, dec_values_lin] = svmpredict(Linear', testDataSVM', model );
[predicted_label_cent, accuracy_cent, dec_values_cent] = svmpredict(centroid_clustering', testDataSVM', model );

predicted_label_KNN
predicted_label_lin
predicted_label_cent
%testData = transpose(testData)

%testDataXY = [final;testData];

%csvwrite('outputFile.txt',testDataXY);

end

