set.seed(200591)
myDummyData <- data.frame(var1 = rnorm(10000, 0, 4)
                          , var2 = rpois(10000, 3)
                          , var3 = rbinom(10000, 1, 0.2)
                          , var4 = runif(10000, 3, 7)
                          , var5 = rbinom(10000, 1, 0.1))
myDummyData$var6 <- log(myDummyData$var2+10)^5/100
myDummyData$var7 <- runif(10000, min = 25, max = 35)*myDummyData$var5
myDummyData$actuals <- with(myDummyData, sqrt(var1^2 + 3*var2)*(1+var3) + var4*(var6-2)/2 + var7)
myDummyData$target <- myDummyData$actuals + rnorm(10000, 0, 1.5)

hist(myDummyData$target, breaks = 50)
summary(myDummyData)
head(myDummyData)


min_rmse <- sqrt(mean((myDummyData$target - myDummyData$actuals)^2))

myDummyData$mean <- mean(myDummyData$target)
max_rmse <- sqrt(mean((myDummyData$target - myDummyData$mean)^2))

featureCols <- grep("var", colnames(myDummyData))
xDf <- myDummyData[,featureCols]
x   <- as.matrix(xDf)
yDf <- myDummyData[,c("target")]
y   <- as.matrix(yDf)

# xg-boost --------------------------------------

#install.packages("xgboost")
library (xgboost)

myDummyDataM <- xgb.DMatrix(data = as.matrix(myDummyData[,featureCols])
                            , label = as.matrix(myDummyData[,c("target")]))

params <- list(booster = "gbtree"
               , objective = "reg:linear"
               , eta=0.3
               , gamma=0
               , max_depth=6
               , min_child_weight=1
               , subsample=1
               , colsample_bytree=1)

xgb <- xgb.train(params = params
                , data = myDummyDataM
                , nrounds = 150
                , nfold = 10
                , showsd = T
                , stratified = T
                , print_every_n = 10
                , early_stop_round = 20
                , maximize = F
                , prediction = T
                , metrics = "rmse")


# linear regression -----------------------------

#install.packages("glmnet")
require(glmnet)

glmLmabda <- cv.glmnet(x = x
                       , y = y
                       , type.measure="mse"
                       , nfolds = 10)

lambda <- c(0, seq(from = glmLmabda$lambda.min, to = glmLmabda$lambda.1se, by = 0.01))

myGlm <- glmnet(x = x
              , y = y
              , alpha = 1
              , family="gaussian"
              , lambda = lambda)

coef(myGlm,s=c(0, glmLmabda$lambda.min, glmLmabda$lambda.1se))


# randomForest ----------------------------------

#install.packages("randomForest")
require(randomForest)
#install.packages("ggplot2")
library(ggplot2)


set.seed(200597)
myDummyData$cvSplit <- round(runif(10000, 0.5, 5.5))
randFSummary <- c()
for (i in 1:5){
  xLoop <- myDummyData[myDummyData$cvSplit != i, featureCols]
  yLoop <- myDummyData[myDummyData$cvSplit != i, c("target")]
  xLoopTest <- myDummyData[myDummyData$cvSplit == i, featureCols]
  aLoopTest <- myDummyData[myDummyData$cvSplit == i, c("target")]
  print(i)
  for (j in 1:ceiling(length(featureCols)/2)){
    randF <- randomForest(x = xLoop
                          , y = yLoop
                          , mtry = j)
    rmseCalc <- sqrt(mean((aLoopTest - predict(randF, newdata = xLoopTest))^2))
    randFSummary <- rbind(randFSummary, c(i, j, rmseCalc))
    print(j)
    print(randFSummary)
  }
}
randFSummary <- as.data.frame(randFSummary)
colnames(randFSummary) <- c("foldNo", "mtry", "rmse")
randFSummary$foldNo <- as.factor(randFSummary$foldNo)
randFSummary$mtry   <- as.factor(randFSummary$mtry)


p <- ggplot(randFSummary, aes(x=mtry, y=rmse, color=mtry)) + geom_boxplot()
p

randF <- randomForest(x = xDf
                      , y = yDf
                      , ntree = 500
                      #, sampsize = 60
                      , mtry = 4
                      , nodesize = 5)

#install.packages("rpart")
library(rpart)
dTree <- rpart(target ~ var1 + var2 + var3 + var4 + var5 + var6 + var7
                     , data = myDummyData
                     , method = "anova"
                     , control = rpart.control(maxdepth = 4, cp = 0.001)
                     )

# Plot the tree.
plot(dTree)
text(dTree, use.n = TRUE)


# Neural Nets -----------------------------------

#install.packages("keras")
library(keras)
#install_keras()

set.seed(200601)
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = "relu", input_shape = length(featureCols)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1, activation = "linear")#"softmax")

model %>% compile(
  loss = "mean_squared_error",
  optimizer = optimizer_adam(),
  metrics = c("mse")
)

history <- model %>% fit(
  x, y, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)


# EVALUATION ----------------------------------------------

set.seed(200593)
myTestData <- data.frame(var1 = rnorm(10000, 0, 4)
                          , var2 = rpois(10000, 3)
                          , var3 = rbinom(10000, 1, 0.2)
                          , var4 = runif(10000, 3, 7)
                          , var5 = rbinom(10000, 1, 0.1))
myTestData$var6 <- log(myTestData$var2+10)^5/100
myTestData$var7 <- runif(10000, min = 25, max = 35)*myTestData$var5
myTestData$actual <- with(myTestData
                          , sqrt(var1^2 + 3*var2)*(1+var3) + var4*(var6-2)/2 + var7) + rnorm(10000, 0, 1.5)

rmse <- function(predictionColName){
  rmse <- sqrt(mean((myTestData$actual - myTestData[,c(predictionColName)])^2))
  rmse
}

newx <- as.matrix(myTestData[,featureCols])

myTestData$xgbPred   <- predict(xgb,   newdata = newx)
myTestData$glm0Pred  <- predict(myGlm, newx = newx, s=0)
myTestData$glm1Pred  <- predict(myGlm, newx = newx, s=glmLmabda$lambda.min)
myTestData$glm2Pred  <- predict(myGlm, newx = newx, s=mean(lambda))
myTestData$glm3Pred  <- predict(myGlm, newx = newx, s=max(lambda))
myTestData$randFPred <- predict(randF, newdata = newx)
myTestData$dTreePred <- predict(dTree, newdata = myTestData[,featureCols])
myTestData$nnPred    <- model %>% predict(newx)


head(myTestData)

myModels <- colnames(myTestData)[grep("Pred", colnames(myTestData))]

myModelEval <- c()
for (i in myModels){
  myModelEval <- rbind(myModelEval, c(i, rmse(i)))
}
plot(c(min_rmse, myModelEval[,2], max_rmse))

myModelEval <- rbind(c("min_rmse", min_rmse)
                     , myModelEval
                     , c("max_rmse", max_rmse))

myModelEval

plot(myTestData$actual, myTestData$randFPred)
plot(myTestData$actual, myTestData$var7)
