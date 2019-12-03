library(caret)
animal_data <-scat
str(scat)

head(animal_data[, 1:10])

sum(is.na(animal_data$Species))
#Q1.  Takes Species as the target value and convert into numerical  
animal_data$Species <-ifelse(animal_data$Species =="bobcat", 1, ifelse(animal_data$Species =="coyote", 2, 3))
target<-animal_data$Species
str(target)

#Q2.  Remove the Month, Year, Site, Location features

animal_data$Month<-NULL
animal_data$Year<-NULL
animal_data$Site<-NULL
animal_data$Location<-NULL
str(animal_data)
#Q3. Check if any values are null. If there are, impute missing values using KNN.

sum(is.na(animal_data))
library('RANN')
preProcValues <- preProcess(animal_data, method = c("knnImpute"))
train_processed <- predict(preProcValues, animal_data)
sum(is.na(train_processed))
str(train_processed)

#Q4. Converting every categorical variable to numerical.
id<-animal_data$Species
id
dmy <- dummyVars(" ~ .", data = train_processed ,fullRank = T)
train_transformed <- data.frame(predict(dmy, newdata = train_processed))
str(train_transformed)
train_transformed$Species<-NULL
train_transformed$Species<-id
str(train_transformed)
#Q5:With a seed of 100, 75% training, 25% testing . Build the following models: randomforest, neural
#net, naive bayes and GBM.

set.seed(100)

#seperating Testing and Training Data 
index <- createDataPartition(train_transformed$Species , p=0.75, list=FALSE)
trainSet <- train_transformed[ index,]
testSet <- train_transformed[-index,]
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE)
outcomeName<-'Species'
predictors<-names(trainSet)[!names(trainSet) %in% outcomeName]


#Loan_Pred_Profile <- rfe(trainSet[,predictors], trainSet[,outcomeName],
#                        rfeControl = control)
#Loan_Pred_Profile
#predictors<-c("CN", "Mass", "segmented", "d13C", "Diameter")
#Using RandomForest 
#a)model summarization

model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf', importance=T)
print(model_rf)

#b)Ploting Variable of Importance for the predictions 

plot(varImp(object=model_rf), main= "Random Forest -Variable of Importance")
predictions<-predict.train(object=model_rf,testSet[,predictors],type="raw")
table(predictions)
dim(predictions)
testSet["Species"]<-as.factor(testSet[,outcomeName])
dim(testSet["Species"])
testSet["Species"]
#c)confusion matrix
confusionMatrix(predictions,testSet["Species"])


#Using neural net

#a)model summarization

model_rf2<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet')
print(model_rf2)

#b)Ploting Variable of Importance for the predictions 
plot(varImp(object=model_rf2), main= "Neural Network -Variable of Importance")
predictions<-predict.train(object=model_rf2,testSet[,predictors],type="raw")
table(predictions)

#c)confusion matrix
confusionMatrix(predictions,testSet[,outcomeName])

#Using naive bayes

#a)model summarization
set.seed(100)
model_rf3<-train(trainSet[,predictors],trainSet[,outcomeName],method='nb')
print(model_rf3)

#b)Ploting Variable of Importance for the predictions 
plot(model_rf3, main= "Neural Network -Variable of Importance")
predictions<-predict.train(object=model_rf3,testSet[,predictors],type="raw")
table(predictions)
predictions
testSet[,outcomeName]<-testSet[,outcomeName].asfactor
#c)confusion matrix
confusionMatrix(predictions,testSet[,outcomeName])

#Using GBM

#a)model summarization

model_rf4<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm')
print(model_rf4)

#b)Ploting Variable of Importance for the predictions 
plot(model_rf4, main= "GBM -Variable of Importance")
predictions<-predict.train(object=model_rf4,testSet[,predictors],type="raw")
table(predictions)

#c)confusion matrix
confusionMatrix(predictions,testSet[,outcomeName])

#Q7 Tune the GBM model using tune length = 20 and: a) print the model summary and b) plot the
#models. (20 points)

fitControl <- trainControl(
  method = "repeatedcv",
  number = 20,
  repeats = 20)

modelLookup(model='gbm')

grid <- expand.grid(n.trees=c(10,20,50,100,500,1000),shrinkage=c(0.01,0.05,0.1,0.5),n.minobsinnode = c(3,5,10),interaction.depth=c(1,5,10))
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneGrid=grid)

# a) summarizing the model
print(model_gbm)
# b) Plot the models
plot(model_gbm)

#8. Using GGplot and gridExtra to plot all variable of importance plots into one single plot. (10
#points)
library("ggplot2")
library("maps")
library("magrittr")

test_data <- data.frame("RandomForest" =model_rf, "NNet" = model_rf2, "Naive_B" = model_rf3, "GBM" =model_rf4)
ggplot(test_data) 

#9. Which model performs the best? and why do you think this is the case? Can we accurately
#predict species on this dataset? (10 points)
