###### EECS 6893 final Project #######

###read the meta data
meta <- read.csv("ColumnsType.csv")
color <- c(rep("orange",14),rep("green",32),rep("blue",20))

######training
train <- read.csv("Merged_clinbioimg.csv")
train_norm_final <- train[,2:68]
train_norm_final$label <- ifelse(train_norm_final$DX.bl== "CN","CN","AD")

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
rfModel2 <- randomForest(DX.bl ~. , data = train_asd, type = "classification",ntree=600)
rfModel3 <- randomForest(DX.bl ~. , data = train_asd, type = "classification",ntree=700)
rfModel4 <- randomForest(DX.bl ~. , data = train_asd, type = "classification",ntree=800)
print(rfModel4)
testRF <- predict(rfModel,test_asd[,1:66],type = "prob")
testRF2 <- predict(rfModel2,test_asd[,1:66],type = "prob")
testRF3 <- predict(rfModel3,test_asd[,1:66],type = "prob")
testRF4 <- predict(rfModel4,test_asd[,1:66],type = "prob")

importance(rfModel)

##### logistic regression
library(caret)
library(glmnet)
glmModel <- glm(DX.bl ~.,family=binomial(link="logit"),data = train_asd)
testLR <- predict(glmModel,test_asd[,1:66])

#### naive bayes classifier
nbModel <- naiveBayes(train_asd[,1:66], as.factor(train_asd[,67]))
testNB <- predict(nbModel,test_asd[,1:66])

### calculate accuracy
c_test <- table(predict(rfModel, test_asd[,1:66]),test_asd[,67])
ac_test <- (c_test[1,1]+c_test[2,2])/nrow(test_asd)


######
library(ggplot2)
library(reshape2)

fea_im <- read.table("importance.txt",sep=" ")

temp <- train_norm_final


cormat <- cor(temp[,1:66],method = "spearman")
melted_cormat <- melt(cormat)
#color <- c(rep("orange",3),rep("green",10),rep("blue",8))
color <- c(rep("orange",14),rep("green",32),rep("blue",20))

# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}

upper_tri <- get_upper_tri(cormat)
melted_cormat <- melt(upper_tri, na.rm = TRUE)
colnames(melted_cormat) <- c("Features.x","Features.y","value")

library(ggplot2)
pdf("features_correlation.pdf")
ggplot(data = melted_cormat, aes(Features.x, Features.y, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Spearman\nCorrelation") +theme_minimal()+ 
                       theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 4, hjust = 1,color = color),
                             axis.text.y = element_text( size = 5,color=color))+
  coord_fixed()
dev.off()


###### plot ROC
library(pROC)
pdf("roc_curve_specific_feautures.pdf")
plot(roc(test_asd$label, as.numeric(testLR)),col="green", lwd=2)
lines(roc(test_asd$label, as.numeric(testRF[,1])),col="blue", lwd=2)
lines(roc(test_asd$label, as.numeric(testSVM)),col="red", lwd=2)
lines(roc(test_asd$label, as.numeric(testNB)),col="purple", lwd=2)
legend("bottomright", legend=c("Random forest (0.6747)", "Logistic regression (0.6945)","SVM kernal (0.573)","Naive Bayes (0.6011)"), 
       col=c("blue", "green","red","purple"), lwd=2)
dev.off()



library(pROC)
pdf("roc_curve_specific_feautures.pdf")
plot(roc(test_asd$DX.bl, as.numeric(testRF1[,1])),col="green", lwd=2)
lines(roc(test_asd$DX.bl, as.numeric(testRF2[,1])),col="blue", lwd=2)
lines(roc(test_asd$DX.bl, as.numeric(testRF3[,1])),col="red", lwd=2)
lines(roc(test_asd$DX.bl, as.numeric(testRF4[,1])),col="purple", lwd=2)
legend("bottomright", legend=c("Random forest (500)", "Random forest (600)","Random forest (700)","Random forest (800)"), 
       col=c("blue", "green","red","purple"), lwd=2)
dev.off()

library(Rtsne)
set.seed(42)
train_norm_final <- train_norm_final[!duplicated(train_norm_final), ]
tsne_result = Rtsne(train_norm_final[,1:66], theta=0,perplexity=30,max_iter=5000)
tsne_result2 = Rtsne(train_norm_final[,1:66], theta=0,perplexity=5,max_iter=5000)

result <- data.frame(X = tsne_result2$Y[,1], Y =  tsne_result2$Y[,2],label=train_norm_final[,67])

labels <- c("AD" = "red", "LMCI" = "yellow","EMCI"="green","CN"="blue") 
p = ggplot(result) +
  geom_point(aes(x=X, y=Y, color=label), size=.4) +
  scale_color_manual(values=labels, name = "Disease Stage") +
  labs(x='tsne1', y='tsne2', title='TSNE result') +
  theme(legend.position = "right", legend.title = element_text(size=7), 
        axis.title = element_text(size=8),title = element_text(size=10),
        legend.text = element_text(size=5),legend.key.size = unit(.5, "cm"))
ggsave('adni_tsne.pdf', plot=p, width=width, height=height)


fea_im <- read.csv("feature_importance.csv")

color <- c(rep("orange",14),rep("green",32),rep("blue",20))
pdf("gene_fe.pdf")
p <-ggplot(fea_im[1:14,], aes(Feature, MeanDecreaseGini))
p + geom_bar(stat = "identity") +theme(axis.text.x = element_text(angle = 90))
dev.off()

as.character(fea_im[which(fea_im$MeanDecreaseGini>2.5),]$Feature)
