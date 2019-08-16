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
dim(myDummyDataM)

params <- list(booster = "gbtree"
               , objective = "reg:linear"
               , eta=0.3
               , gamma=0
               , max_depth=6
               , min_child_weight=1
               , subsample=1
               , colsample_bytree=1)

xgbcv <- xgb.cv(params = params
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

myDummyData$pred <- xgbcv$pred
plot(myDummyData$target, myDummyData$pred)
xgb_rmse <- sqrt(mean((myDummyData$target - myDummyData$pred)^2))
xgb_rmse

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

lambda <- seq(from = glmLmabda$lambda.min, to = glmLmabda$lambda.1se, by = 0.01)

myGlm <- glmnet(x = x
              , y = y
              , family="gaussian"
              , lambda = lambda)
myGlm$beta
coef(myGlm,s=mean(lambda)) 
myDummyData$glm0 <- predict(myGlm, newx=x, s=0)
myDummyData$glm1 <- predict(myGlm, newx=x, s=min(lambda))
myDummyData$glm2 <- predict(myGlm, newx=x, s=mean(lambda))
myDummyData$glm3 <- predict(myGlm, newx=x, s=max(lambda))

plot(myDummyData$target, myDummyData$glm0)
glm0_rmse <- sqrt(mean((myDummyData$target - myDummyData$glm0)^2))
glm1_rmse <- sqrt(mean((myDummyData$target - myDummyData$glm1)^2))
glm2_rmse <- sqrt(mean((myDummyData$target - myDummyData$glm2)^2))
glm3_rmse <- sqrt(mean((myDummyData$target - myDummyData$glm3)^2))
c(glm0_rmse, glm1_rmse, glm2_rmse, glm3_rmse)
