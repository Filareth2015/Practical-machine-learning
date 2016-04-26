# Prediction-Assignment-Writeup
Shebloff  
26 april 2016  



## R Markdown

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. The goal of your project is to predict the manner in which they did the exercise.

## Cleaning and preparing Data
First erasing empty and unrelevant columns

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.2.5
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.2.5
```

```r
set.seed(321)

train <- read.csv("pml-training.csv", stringsAsFactors=FALSE)
train$classe <- as.factor(train$classe)
train <- train[,-c(1:7)]
train <- train[,-nearZeroVar(train)]
```

Using KNN method. PCA to reduce features.


```r
inTrain <- createDataPartition(y=train$classe, p=0.7, list=FALSE)
train <- train[inTrain,]
test <- train[-inTrain,]

l<-length(train)
preObj <- preProcess(train[,-l],method=c("center","scale","knnImpute","pca"), thresh=0.9)
clean_data <- predict(preObj,train[,-l])
```

##Prediction
The accuracy of KNN is 0.9643, which would give us an out of sample error of 1-0.9643= 0.0357

```r
modelFit <- train(train$classe ~.,data=clean_data, method="knn")
testresult <- predict(preObj, test[,-length(test)])
confusionMatrix(test$classe, predict(modelFit,testresult))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1166    7    9    5    0
##          B   13  746   27    4    3
##          C    5   10  692    8    3
##          D    5    0   24  671    1
##          E    3    8    5    7  698
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9643          
##                  95% CI : (0.9582, 0.9698)
##     No Information Rate : 0.2893          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9548          
##  Mcnemar's Test P-Value : 0.0002696       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9782   0.9676   0.9141   0.9655   0.9901
## Specificity            0.9928   0.9860   0.9923   0.9912   0.9933
## Pos Pred Value         0.9823   0.9407   0.9638   0.9572   0.9681
## Neg Pred Value         0.9911   0.9925   0.9809   0.9930   0.9979
## Prevalence             0.2893   0.1871   0.1837   0.1687   0.1711
## Detection Rate         0.2830   0.1811   0.1680   0.1629   0.1694
## Detection Prevalence   0.2881   0.1925   0.1743   0.1701   0.1750
## Balanced Accuracy      0.9855   0.9768   0.9532   0.9784   0.9917
```

##And predictiong on testing data


```r
test <- read.csv("pml-testing.csv", stringsAsFactors=FALSE)
test <- test[,names(test) %in% names(train)]

testresult <- predict(preObj, test)
predict_result <- predict(modelFit, testresult)
predict_result
```

```
##  [1] B A C A A E D B A A B C B A E A A B B B
## Levels: A B C D E
```

