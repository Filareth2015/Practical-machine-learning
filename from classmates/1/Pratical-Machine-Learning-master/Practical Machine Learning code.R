#Packages, Library, Seed -->Need to be installed
library(caret) #Machine learning Library
library(ggplot2) 
library(kernlab)
library(lattice)
library(randomForest) #Random forest for classigication and regression
library(rpart) #Regressive partitioning and Regression trees
library(rpart.plot) # Decision Tree Plot
library(e1071)
library(rattle)
library(gbm)
library(survival)
library(caTools)
library(ranger)
library(plyr) #Boosted Regression
library(data.table)
set.seed(12345) # For reproducibility

################################################################################
#Import .csv training and testing dataset | Preliminary clearning
## Data Collection for Training set: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
## Data collection for Test set: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
## After saving both data sets into my working directory, some missing values are coded as string "#DIV/0" or "" or "NA" - these will be cahnged to NA
## Both data sets contain columns with all missing values - there will be deleted

## Loading the training data set into my R section replacing all missing with "NA"
trainingset <- read.csv("~/R/pml-training.csv", na.strings=c("NA","#DIV/0!", ""))
str(trainingset) ## to check if there is still "#DIV/0!" or "" in the trainingset
## Loading the testing data set                        
testingset <- read.csv("~/R/pml-testing.csv", na.strings=c("NA","#DIV/0!",""))
str(testingset) ## to check if there is still "#DIV/0!" or "" in the testingset

## Check dimensions for number of variables and observations
dim(trainingset)
dim(testingset)

## Delete columns with all missing values
trainingset<-trainingset[,colSums(is.na(trainingset))==0]
testingset<-testingset[,colSums(is.na(testingset))==0]

##Some variables are irrelevant to our current project: user_name, raw_timestap_part_,2 cvtd_timestamp, new_window,and num_window (column 1 to 7) --> They can be delete
trainingset<-trainingset[,-c(1:7)]
testingset<-testingset[,-c(1:7)]

##Check the dimension of the new datasets
dim(trainingset) 
##Result: [1] 19622 (observations) 53 (variables)
dim(testingset) 
##Result: [1] 20 (observations) 53 (variables)
head(trainingset)
head(testingset)

################################################################################

#Partitioning the training data set to allow cross-validation
##In order to perform cross-validation, the training dataset is partitionned into 2 sets: subTraining(75%) and subTest(25%)
##The cross-validation technique used is the random subsampling.
subsamples<-createDataPartition(y=trainingset$classe,p=0.75,list=FALSE)
subTraining<-trainingset[subsamples,]
subTesting<-trainingset[-subsamples,]
dim(subTraining)
## Result: [1] 14718    53
dim(subTesting)
## Result: [1] 4904   53
head(subTraining)
head(subTesting)

#Look at the Data
##In Wallace Ugulino et al., 2012 study, the variable "classe" contains 5 levels: A, B, C, D, E, assessing how well they do in a particular activity
## Plotting the variable "classe" enable to see the frequency of each levels in the subTraining dataset and compare one another
plot(subTraining$classe, col="blue", main="Bar plot of levels of the variable classe within the subTraining dataset",xlab="classe levels", ylab="Frequency")


################################################################################

#First Prediction Model using the Decision Tree.
model1<-rpart(classe~., data=subTraining, method="class")
##Predicting:
prediction1<-predict(model1,subTesting, type="class")
##Plot the Decision Tree
rpart.plot(model1,main="Classification Tree", extra=102,under=TRUE,faclen=0)

## Test results on our subTesting data set using the subTraining parameter and determine the accuracy
confusionMatrix(prediction1, subTesting$classe) # Results: Accuracy=0.7229
######RESULT######
## Confusion Matrix and Statistics

## Reference
## Prediction   
##      A    B    C    D    E
## A 1260  156   33   40   23
## B   52  555   73   52   52
## C   24  136  575   83   95
## D   40   33  150  513   89
## E   19   69   24  116  642

## Overall Statistics

## Accuracy : 0.7229          
## 95% CI : (0.7101, 0.7354)
## No Information Rate : 0.2845          
## P-Value [Acc > NIR] : < 2.2e-16       

## Kappa : 0.6486          
## Mcnemar's Test P-Value : < 2.2e-16       

##Statistics by Class:

## Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9032   0.5848   0.6725   0.6381   0.7125
## Specificity            0.9282   0.9421   0.9165   0.9239   0.9430
## Pos Pred Value         0.8333   0.7079   0.6298   0.6218   0.7379
## Neg Pred Value         0.9602   0.9044   0.9298   0.9287   0.9358
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2569   0.1132   0.1173   0.1046   0.1309
## Detection Prevalence   0.3083   0.1599   0.1862   0.1682   0.1774
## Balanced Accuracy      0.9157   0.7635   0.7945   0.7810   0.8278

################################################################################
#Second Prediction Model using the Random Forest.
model2<-randomForest(classe~., data=subTraining, method="class") #Longer Technique
##Predicting:
prediction2<-predict(model2, subTesting, type="class")

## Test results on our subTesting data set using the subTraining parameter and determine the accuracy
confusionMatrix(prediction2, subTesting$classe) #Results: Accuracy=0.9949 !

######RESULT######
## Confusion Matrix and Statistics

## Reference
## Prediction

##      A    B    C    D    E
## A 1395    6    0    0    0
## B    0  939    2    0    0
## C    0    4  852    7    1
## D    0    0    1  797    4
## E    0    0    0    0  896

## Overall Statistics

## Accuracy : 0.9949          
## 95% CI : (0.9925, 0.9967)
## No Information Rate : 0.2845          
## P-Value [Acc > NIR] : < 2.2e-16       

## Kappa : 0.9936          
## Mcnemar's Test P-Value : NA              

## Statistics by Class:

## Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9895   0.9965   0.9913   0.9945
## Specificity            0.9983   0.9995   0.9970   0.9988   1.0000
## Pos Pred Value         0.9957   0.9979   0.9861   0.9938   1.0000
## Neg Pred Value         1.0000   0.9975   0.9993   0.9983   0.9988
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1915   0.1737   0.1625   0.1827
## Detection Prevalence   0.2857   0.1919   0.1762   0.1635   0.1827
## Balanced Accuracy      0.9991   0.9945   0.9968   0.9950   0.9972

################################################################################
#Third Prediction Model using the K-Bearest Neightbors Model.
model3<- train(classe ~ ., data=subTraining, method = "knn", trControl = trainControl(method = "adaptive_cv"))
##Predicting:
prediction3<-predict(model3, subTesting)

## Test results on our subTesting data set using the subTraining parameter and determine the accuracy
confusionMatrix(prediction3, subTesting$classe) #Results: Accuracy=0.9162

######RESULT######
## Confusion Matrix and Statistics

## Reference
## Prediction    A    B    C    D    E
## A 1337   45   11   10    7
## B   19  816   23    5   34
## C   22   43  791   47   19
## D   12   23   18  732   24
## E    5   22   12   10  817

## Overall Statistics

## Accuracy : 0.9162          
## 95% CI : (0.9081, 0.9238)
## No Information Rate : 0.2845          
## P-Value [Acc > NIR] : < 2.2e-16       

## Kappa : 0.894           
## Mcnemar's Test P-Value : 2.859e-08       

## Statistics by Class:

## Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9584   0.8599   0.9251   0.9104   0.9068
## Specificity            0.9792   0.9795   0.9676   0.9812   0.9878
## Pos Pred Value         0.9482   0.9097   0.8579   0.9048   0.9434
## Neg Pred Value         0.9834   0.9668   0.9839   0.9824   0.9792
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2726   0.1664   0.1613   0.1493   0.1666
## Detection Prevalence   0.2875   0.1829   0.1880   0.1650   0.1766
## Balanced Accuracy      0.9688   0.9197   0.9464   0.9458   0.9473

################################################################################
#Fourth Prediction Model using the logit boosting.
model4<- train(classe ~ ., data=subTraining, method = "LogitBoost")
##Predicting:
prediction6<-predict(model4, subTesting)

## Test results on our subTesting data set using the subTraining parameter and determine the accuracy
confusionMatrix(prediction4, subTesting$classe) #Results: 0.8924 

######RESULT######
## Confusion Matrix and Statistics

## Reference
## Prediction   
##      A    B    C    D    E
## A 1303   85   11   16   10
## B   15  667   58   12   29
## C    6   42  624   59   31
## D    6    4   22  552   25
## E    2   16    5   11  709

## Overall Statistics

## Accuracy : 0.8924          
## 95% CI : (0.8827, 0.9015)
## No Information Rate : 0.3083          
## P-Value [Acc > NIR] : < 2.2e-16       

## Kappa : 0.8623          
## Mcnemar's Test P-Value : < 2.2e-16       

## Statistics by Class:

##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9782   0.8194   0.8667   0.8492   0.8818
## Specificity            0.9592   0.9675   0.9617   0.9845   0.9903
## Pos Pred Value         0.9144   0.8540   0.8189   0.9064   0.9542
## Neg Pred Value         0.9900   0.9585   0.9730   0.9736   0.9734
## Prevalence             0.3083   0.1884   0.1667   0.1505   0.1861
## Detection Rate         0.3016   0.1544   0.1444   0.1278   0.1641
## Detection Prevalence   0.3299   0.1808   0.1764   0.1410   0.1720
## Balanced Accuracy      0.9687   0.8934   0.9142   0.9168   0.9361

################################################################################
#Comparaison of models
##The model2 built from the Random Forest algorithm has a higher accuracy (=0.9949 with CI:(0.9925, 0.9967)) compared to the Decision Tree algorithm (=0.7229 with CI:(0.7101, 0.7354))
## or the K-Bearest Neightbors Model (=0.9162 with CI(0.9081, 0.9238))
##Hence, the Random Forest model2 is choosen to be applied to the Testing data set
##The expected out-of-sample error is estimated at 0.5%. 
##The expected out-of-sample error is calculated as 1-accuracy for predictions made agains the cross-validation set.
##Our Test data set comprises 20 observations. With an accuracy >99% on our cross-validation data, just a very few or none of the test samples will be missclassified.

################################################################################
#Submission
##Predict outcome levels on the original testing dataset using Random Forest Algorithm
predictfinal<-predict(model2, testingset, type="class")
predictfinal
######RESULT######
## 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
## B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E


##Write files for submission
pml_write_files=function(x){
  n=length(x)
  for (i in 1:n) {
    filename=paste0("problem_id_", i, ".txt")
    write.table(x[i],filename, quote=FALSE, row.names=FALSE, col.names=FALSE)
  }
}
pml_write_files(predictfinal)