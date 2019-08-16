set.seed(200591)
myDummyData <- data.frame(var1 = rnorm(10000, 0, 4)
                          , var2 = rpois(10000, 3)
                          , var3 = rbinom(10000, 1, 0.2)
                          , var4 = runif(10000, 3, 7))
myDummyData$var5 <- log(myDummyData$var2+10)^5/100
myDummyData$actuals <- with(myDummyData, sqrt(var1^2 + 3*var2)*(1+var3) + var4*(var5-2)/2)
myDummyData$target <- myDummyData$actuals + rnorm(10000, 0, 1.5)

summary(myDummyData)
head(myDummyData)
hist(myDummyData$target)


min_rmse <- sqrt(mean((myDummyData$target - myDummyData$actuals)^2))

myDummyData$mean <- mean(myDummyData$target)
max_rmse <- sqrt(mean((myDummyData$target - myDummyData$mean)^2))

featureCols <- grep("var", colnames(myDummyData))

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

x <- as.matrix(myDummyData[,featureCols])
y <- as.matrix(myDummyData[,c("target")])

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

set.seed(200597)
randF <- randomForest(x = myDummyData[,featureCols]
                      , y = myDummyData[,c("target")])


# randomForest ----------------------------------

install.packages("keras")
library(keras)
install_keras()

set.seed(200601)

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = "relu", input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = "softmax")


# EVALUATION ----------------------------------------------

set.seed(200593)

myTestData <- data.frame(var1 = rnorm(10000, 0, 4)
                          , var2 = rpois(10000, 3)
                          , var3 = rbinom(10000, 1, 0.2)
                          , var4 = runif(10000, 3, 7))
myTestData$var5 <- log(myTestData$var2+10)^5/100
myTestData$actual <- with(myDummyData, sqrt(var1^2 + 3*var2)*(1+var3) + var4*(var5-2)/2) + rnorm(10000, 0, 1.5)


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
myTestData$randFPred <- predict(randF, newx = newx)

head(myTestData)

myModels <- colnames(myTestData)[grep("Pred", colnames(myTestData))]

myModelEval <- c()
for (i in myModels){
  myModelEval <- rbind(myModelEval, c(i, rmse(i)))
}
plot(c(min_rmse, myModelEval[,2], max_rmse))

myModelEval

plot(myTestData$actual, myTestData$randFPred)
