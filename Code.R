library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(randomForest)

set.seed(12345)

trainingURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training <- read.csv(trainingURL, na.strings=c("NA",""), header=TRUE)
testing <- read.csv(testURL, na.strings=c("NA",""), header=TRUE)

indexForNA_training <- apply(training,2,function(x) {sum(is.na(x))}) 
training <- training[,which(indexForNA_training == 0)]
indexForNA_testing <- apply(testing,2,function(x) {sum(is.na(x))}) 
testing <- testing[,which(indexForNA_testing == 0)]


training$classe <- as.factor(training$classe) 

numericCol <- which(lapply(training, class) %in% "numeric")

preObj <-preProcess(training[,numericCol],method=c('knnImpute', 'center', 'scale'))
trainPreProcessed <- predict(preObj, training[,numericCol])
trainPreProcessed$classe <- training$classe
testingPreProcessed <-predict(preObj,testing[,numericCol])

nzvTraining <- nearZeroVar(trainPreProcessed,saveMetrics=TRUE)
trainPreProcessed <- trainPreProcessed[,nzvTraining$nzv==FALSE]
nzvTesting <- nearZeroVar(testingPreProcessed,saveMetrics=TRUE)
testingPreProcessed <- testingPreProcessed[,nzvTesting$nzv==FALSE]

inTrain = createDataPartition(trainPreProcessed$classe, p = 3/4, list=FALSE)
trainingPart = trainPreProcessed[inTrain,]
testingPart = trainPreProcessed[-inTrain,]

decisiontree <- train(classe~.,method="rpart", data=trainingPart)
fancyRpartPlot(decisiontree$finalModel)

predictions <- predict(decisiontree,newdata = testingPart)
confusionMatrix(testingPart$classe, predictions)
                       

modFit <- train(classe ~.,
                method="rf", 
                data=trainingPart, 
                trControl=trainControl(method='cv'), 
                number=5, 
                allowParallel=TRUE )

modFit

trainingPartPrediction <- predict(modFit, trainingPart)
confusionMatrix(trainingPartPrediction, trainingPart$classe)

testingPartPrediction <- predict(modFit, testingPart)
confusionMatrix(testingPartPrediction, testingPart$classe)

testingPrediction <- predict(modFit, testingPreProcessed)
testingPrediction
