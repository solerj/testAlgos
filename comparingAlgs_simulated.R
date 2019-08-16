set.seed(200591)
myDummyData <- data.frame(var1 = rnorm(10000, 0, 4)
                          , var2 = rpois(10000, 3)
                          , var3 = rbinom(10000, 1, 0.2))
myDummyData$target <- with(myDummyData, sqrt(var1^2 + 3*var2)*(1+var3)) + rnorm(10000, 0, 1.5)

summary(myDummyData)
head(myDummyData)
hist(myDummyData$target)

myDummyData$actuals <- with(myDummyData, sqrt(var1^2 + 3*var2)*(1+var3))
min_rmse <- sqrt(mean((myDummyData$target - myDummyData$actuals)^2))
min_rmse

myDummyData$mean <- mean(myDummyData$target)
max_rmse <- sqrt(mean((myDummyData$target - myDummyData$mean)^2))
max_rmse

# xg-boost --------------------------------------

#install.packages("xgboost")
library (xgboost)

myDummyDataM <- xgb.DMatrix(data = as.matrix(myDummyData[,c("var1", "var2", "var3")])
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

x <- as.matrix(myDummyData[,1:3])
y <- as.matrix(myDummyData[,c("target")])

glmLmabda <- cv.glmnet(x = x
                       , y = y
                       #, lambda
                       , type.measure="mse"
                       , nfolds = 10)

lambda <- c(0, seq(from = glmLmabda$lambda.min, to = glmLmabda$lambda.1se, by = 0.01))

myGlm <- glmnet(x = x
              , y = y
              , family="gaussian"
              , lambda = lambda)

coef(myGlm,s=mean(lambda)) 



plot(myDummyData$target, myDummyData$glm0)

# EVALUATION ----------------------------------------------

set.seed(200593)
myTestData <- data.frame(var1 = rnorm(10000, 0, 4)
                          , var2 = rpois(10000, 3)
                          , var3 = rbinom(10000, 1, 0.2))
myTestData$actual <- with(myTestData, sqrt(var1^2 + 3*var2)*(1+var3)) + rnorm(10000, 0, 1.5)

rmse <- function(predictionColName){
  rmse <- sqrt(mean((myTestData$actual - myTestData[,c(predictionColName)])^2))
  rmse
}

newx <- as.matrix(myTestData[,c("var1", "var2", "var3")])

myTestData$xgbPred  <- predict(xgb,   newdata = newx)
myTestData$glm0Pred <- predict(myGlm, newx = newx, s=0)
myTestData$glm1Pred <- predict(myGlm, newx = newx, s=glmLmabda$lambda.min)
myTestData$glm2Pred <- predict(myGlm, newx = newx, s=mean(lambda))
myTestData$glm3Pred <- predict(myGlm, newx = newx, s=max(lambda))


myModels <- colnames(myTestData)[grep("Pred", colnames(myTestData))]

myModelEval <- c()
for (i in myModels){
  myModelEval <- rbind(myModelEval, c(i, rmse(i)))
}
plot(c(min_rmse, myModelEval[,2], max_rmse))
