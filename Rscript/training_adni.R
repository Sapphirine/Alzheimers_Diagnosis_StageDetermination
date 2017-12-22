train <- read.csv("Merged_clinbioimg.csv")
train_norm_final <- train[,2:68]
train_norm_final$label <- ifelse(train_norm_final$DX.bl== "CN","CN","AD")

### traing
train_norm_final_asd <- train_norm_final[which(train_norm_final$DX.bl=="AD"),]
train_norm_final_c <- train_norm_final[train_norm_final$DX.bl=="CN",]


n1<- nrow(train_norm_final_asd)
trainIndex1 = sample(1:n1, size = round(0.8*n1), replace=FALSE)
train1 = train_norm_final_asd[trainIndex1,]
test1 <- train_norm_final_asd[-trainIndex1,]

n2<- nrow(train_norm_final_c)
trainIndex2 = sample(1:n2, size = round(0.8*n2), replace=FALSE)
train2 = train_norm_final_c[trainIndex2,]
test2 <- train_norm_final_c[-trainIndex2,]

train_asd <- rbind(train1,train2)
test_asd <- rbind(test1,test2)
train_asd <- train_asd[,-67]
test_asd <- test_asd[,-67]

train_asd$label <- factor(train_asd$label)
test_asd$label <- factor(test_asd$label)

train_asd$DX.bl <- factor(train_asd$DX.bl)
test_asd$DX.bl <- factor(test_asd$DX.bl)

###### svm model
library(e1071)
svmModel <- svm(DX ~ ., data = train_asd,kernel="radial")
testSVM <- predict(svmModel,test_asd[,1:66],type = "prob")

train_asd$DX.bl

##### random forest
library(randomForest)
rfModel <- randomForest(label ~. , data = train_asd, type = "classification",ntree=500)
importance(rfModel)
testRF <- predict(rfModel,test_asd[,1:66],type = "prob")