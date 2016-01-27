library(caret)
library(TTR)
library(cluster)



#--------------------------------------------------------------------
#                               FUNCTIONS 
#--------------------------------------------------------------------

checkY <- function(y, num.pred.col, num.weight.col) {
  wmae <- sum(abs(y - num.pred.col)*num.weight.col) /length(num.pred.col)
  return(wmae)
}

getY <- function(model, test.data, ranges.values) {
  y <- predict(model, test.data)
  y <- as.numeric(y)
  for(i in 1:length(ranges.values)) {
    y[y == i] <- ranges.values[i]
  }
  return(y)
}

getWmae <- function(model, test.data, num.pred.col, num.weight.col, ranges.values) {
  y <- predict(model, test.data)
  y <- as.numeric(y)
  for(i in 1:length(ranges.values)) {
    y[y == i] <- ranges.values[i]
  }
  wmae <- sum(abs(y - num.pred.col)*num.weight.col) /length(num.pred.col)
  return(list('wmae'=wmae, 'y'=y))
}

my.stupid.sma <- function(ret, period = 10) {
  l <- c()
  for(i in 1:(length(ret)/period)){
    l <- append(l, mean(ret[(i*period - (period - 1)):(i*period)]))
  }
  return(l)
}



#--------------------------------------------------------------------
#                              END FUNCTIONS 
#--------------------------------------------------------------------



#--------------------------------------------------------------------
#                     Step 1. Load data
#--------------------------------------------------------------------
train.original.data <- read.csv('train.csv')
test.original.data <- read.csv('test.csv')


#--------------------------------------------------------------------
#                     Step 2. Process data
#--------------------------------------------------------------------
train.original.data$is.train <- 1
test.original.data$is.train  <- 0


#--------------------------------------------------------------------
#                     Step 2.a Drop NAs
#--------------------------------------------------------------------
# means <- c()
# tmp <- rbind(train.original.data[, c(1:147, ncol(train.original.data))], test.original.data)
# tmp <- train.original.data[, c(1:147)]
tmp <- test.original.data
for(i in 1:ncol(tmp)){
  # means <- c(means, mean(tmp[,i], na.rm = TRUE))
  # tmp[is.na(tmp[,i]), i] <- mean(tmp[,i], na.rm = TRUE)
  tmp[is.na(tmp[,i]), i] <- means[i]
}

#
#
#


#--------------------------------------------------------------------
#                     Step 2.b Get clusters
#--------------------------------------------------------------------
tmp$clust <- clara(tmp$Feature_7, k = 16, metric = "manhattan")$clustering
tmp$fit.clust <- clara(tmp[, c(2:7, 9:26)], k = 6, metric = "manhattan")$clustering


#--------------------------------------------------------------------
#                     Step 3. Adding Features
#--------------------------------------------------------------------
# CUM SUM
START.POINT = 10
rets <- as.data.frame(t(tmp[, 29:147]))
rets <- rbind(START.POINT, rets)
cs <- as.data.frame(sapply(rets, cumsum))
dt.cutsum <- as.data.frame(t(cs))
colnames(dt.cutsum) <- paste('cumsum_', 1:120, sep='')

#CUM PROD
START.POINT = 1
rets <- as.data.frame(t(tmp[, 29:147]))
rets <- START.POINT + rets
cs <- as.data.frame(sapply(rets, cumprod))
dt.cutsum <- as.data.frame(t(cs))
colnames(dt.cutsum) <- paste('cumprod_', 1:119, sep='')



#--------------------------------------------------------------------
#                     Step 3.a Adding Technical Indicators
#--------------------------------------------------------------------
#SMA
#EMA
#ALMA
#momentum
#RSI
#MACD
sma.t <- as.data.frame(sapply(cs, SMA, n = 16))
dt.sma.16 <- as.data.frame(t(sma.t))
colnames(dt.sma.16) <- paste('SMA.16_', 1:120, sep='')

ema.t <- as.data.frame(sapply(cs, EMA, n = 10))
dt.ema.10 <- as.data.frame(t(ema.t))
colnames(dt.ema.10) <- paste('EMA.10_', 1:120, sep='')

N <- 9
alma.t <- as.data.frame(sapply(cs, ALMA, n = N))
dt.alma.9 <- as.data.frame(t(alma.t))
colnames(dt.alma.9) <- paste('ALMA.6_', 1:(120 - N + 1), sep='')

dt.alma.sma <- dt.alma.9 - dt.sma.16[, N:120]
colnames(dt.alma.sma) <- paste('ALMA-SMA_', 1:(120 - N + 1), sep='')

dt.alma.ema <- dt.alma.9 - dt.ema.10[, N:120]
colnames(dt.alma.ema) <- paste('ALMA-EMA_', 1:(120 - N + 1), sep='')

mom.t <- as.data.frame(sapply(cs, momentum, n = 2))
dt.mom.2 <- as.data.frame(t(mom.t))
colnames(dt.mom.2) <- paste('MOM.2_', 1:120, sep='')

rsi.t <- as.data.frame(sapply(cs, RSI, n = 14))
dt.rsi.14 <- as.data.frame(t(rsi.t))
colnames(dt.rsi.14) <- paste('RSI.14_', 1:120, sep='')

dt.diff.rsi.t <-  as.data.frame(sapply(rsi.t, diff))
dt.diff.rsi.14 <- as.data.frame(t(dt.diff.rsi.t))
colnames(dt.diff.rsi.14) <- paste('diff.RSI.14_', 1:119, sep='')

rsi.t <- as.data.frame(sapply(cs, RSI, n = 25))
dt.rsi.25 <- as.data.frame(t(rsi.t))
colnames(dt.rsi.25) <- paste('RSI.25_', 1:120, sep='')

dt.diff.rsi.t <-  as.data.frame(sapply(rsi.t, diff))
dt.diff.rsi.25 <- as.data.frame(t(dt.diff.rsi.t))
colnames(dt.diff.rsi.25) <- paste('diff.RSI.25_', 1:119, sep='')

macd.t <- as.data.frame(sapply(cs, MACD))
dt.macd.sd <- as.data.frame(t(macd.t))
colnames(dt.macd.sd) <- paste('MACD_', 1:240, sep='')

dt.osma  <- dt.macd.sd[, 1:120] - dt.macd.sd[, 121:240]
colnames(dt.osma) <- paste('OSMA_', 1:120, sep='')

dt.osma.t <- as.data.frame(t(dt.osma))
dt.diff.osma.t <- as.data.frame(sapply(dt.osma.t, diff))
dt.diff.osma <- as.data.frame(t(dt.diff.osma.t))
colnames(dt.diff.osma) <- paste('diff.OSMA_', 1:119, sep='')
rm(dt.osma.t)
rm(dt.diff.osma.t)


#--------------------------------------------------------------------
#                     Step 3.b Adding some my stupid features
#--------------------------------------------------------------------
sma.t <- as.data.frame(sapply(cs, my.stupid.sma, period = 5))
dt.diff.sma.5 <- as.data.frame(t(as.data.frame(sapply(sma.t, diff))))
dt.sma.5 <- as.data.frame(t(sma.t))
colnames(dt.sma.5) <- paste('sma5_', 1:24, sep='')
colnames(dt.diff.sma.5) <- paste('diff_sma.5_', 1:23, sep='')

sma.t <- as.data.frame(sapply(cs, my.stupid.sma, period = 3))
dt.diff.sma.3 <- as.data.frame(t(as.data.frame(sapply(sma.t, diff))))
dt.sma.3 <- as.data.frame(t(sma.t))
colnames(dt.sma.3) <- paste('sma3_', 1:40, sep='')
colnames(dt.diff.sma.3) <- paste('diff_sma.3_', 1:39, sep='')

sma.t <- as.data.frame(sapply(cs, my.stupid.sma, period = 4))
dt.diff.sma.4 <- as.data.frame(t(as.data.frame(sapply(sma.t, diff))))
dt.sma.4 <- as.data.frame(t(sma.t))
colnames(dt.sma.4) <- paste('sma4_', 1:30, sep='')
colnames(dt.diff.sma.4) <- paste('diff_sma.4_', 1:29, sep='')
rm(cs)
rm(rets)
rm(sma.t)
rm(alma.t)
rm(ema.t)
rm(macd.t)
rm(rsi.t)
rm(mom.t)
rm(i)
rm(N)
rm(START.POINT)
rm(dt.diff.rsi.t)
rm(dt.sma.3)
rm(dt.sma.4)
rm(dt.sma.5)

#--------------------------------------------------------------------
#                     Step 4. Binding all features
#--------------------------------------------------------------------
# tmp.ex.1 <- cbind(tmp$Id, tmp$clust, tmp$fit.clust,
#                   dt.cutsum,
#                   dt.sma.16, dt.ema.10, dt.alma.9,
#                   dt.mom.2, dt.rsi.14, dt.rsi.25, dt.macd.sd,
#                   dt.diff.rsi.25, dt.diff.rsi.14)
tmp.ex.1 <- cbind(tmp$Id, dt.cutsum,
                dt.sma.16, dt.ema.10, dt.alma.9,
                dt.mom.2, dt.rsi.14, dt.rsi.25, dt.macd.sd,
                dt.diff.rsi.25, dt.diff.rsi.14)
tmp.ex.1 <- cbind(tmp.ex.1,
                dt.alma.ema, dt.alma.sma)
rm(dt.alma.ema)
rm(dt.alma.sma)
tmp.ex.1 <- cbind(tmp.ex.1, dt.osma)
tmp.ex.1 <- cbind(tmp.ex.1, dt.diff.osma)
rm(dt.osma)
rm(dt.diff.osma)
tmp.ex.1 <- cbind(tmp.ex.1, dt.diff.sma.3, dt.diff.sma.4, dt.diff.sma.5)
rm(dt.diff.sma.3)
rm(dt.diff.sma.4)
rm(dt.diff.sma.5)

tmp.ex.1 <- cbind(tmp.ex.1, tmp[, 2:147])
colnames(tmp.ex.1)[1] <- "Id"
colnames(tmp.ex.1)[2] <- "clust"
colnames(tmp.ex.1)[3] <- "fit.clust"
#colnames(tmp.ex.1)[3] <- "Feature_7.origin"
tmp.ex <- Filter(function(x)!any(is.na(x)), tmp.ex.1)
rm(tmp.ex.1)


#--------------------------------------------------------------------
#                     Step 5. Removing temp vars
#--------------------------------------------------------------------
rm(dt.rsi.25)
rm(cs)
rm(rets)
rm(dt.sma.5)
rm(dt.sma.4)
rm(dt.sma.3)
rm(dt.diff.sma.5)
rm(dt.diff.sma.4)
rm(dt.diff.sma.3)
rm(sma.t)
rm(alma.t)
rm(dt.alma.9)
rm(dt.cutsum)
rm(dt.ema.10)
rm(ema.t)
rm(dt.sma.16)
rm(dt.macd.sd)
rm(dt.mom.2)
rm(dt.rsi.14)
rm(macd.t)
rm(rsi.t)
rm(mom.t)
rm(tmp)
rm(i)
rm(N)
rm(START.POINT)
rm(dt.alma.ema)
rm(dt.alma.sma)
rm(dt.osma)
rm(dt.diff.osma)
rm(dt.diff.rsi.t)
rm(dt.diff.rsi.14)
rm(dt.diff.rsi.25)


# saving all features
save(tmp.ex, file = '3_tech_ana_tmp.ex')

#
# load('3_tech_ana_tmp.ex')

#--------------------------------------------------------------------
#                     Step 6. Divine to train and test
#--------------------------------------------------------------------
test.original.data   <- subset(tmp.ex, tmp.ex$is.train == 0)
train.original.data  <- cbind(subset(tmp.ex, tmp.ex$is.train == 1), train.original.data[, 148:211])
rm(tmp.ex)



#--------------------------------------------------------------------
#                     Step 7.a Learning
#--------------------------------------------------------------------
library(caret)
ranges.p1 <- c(-1, median.y.neg, median.y.pos, 1)
ranges.p2 <- c(-1, median.p2.neg, median.p2.pos, 1)
ranges.121 <- c(-1, -1e-3, -1e-4, -1e-5, 0, 1e-5, 1e-4, 1e-3, 1)

train.original.data$Cut_P1 <- cut(train.original.data$Ret_PlusOne, ranges.p1)
train.original.data$Cut_P2 <- cut(train.original.data$Ret_PlusOne, ranges.p2)


train.index <- createDataPartition(train.original.data$Cut_P1, p = 0.30, list = F)
dt.train <- train.original.data[train.index, ]
dt.test  <- train.original.data[-train.index, ]

dt.train.g1 <- subset(dt.train, dt.train$clust == 1)
dt.train.g2 <- subset(dt.train, dt.train$clust == 2)
dt.train.g3 <- subset(dt.train, dt.train$clust == 3)
dt.train.g4 <- subset(dt.train, dt.train$clust == 4)

dt.test.g1 <- subset(dt.test, dt.test$clust == 1)
# dt.test.g2 <- subset(dt.test, dt.test$clust == 2)
# dt.test.g3 <- subset(dt.test, dt.test$clust == 3)
# dt.test.g4 <- subset(dt.test, dt.test$clust == 4)


#--------------------------------------------------------------------
#                     Step 7.b Fitting model
#--------------------------------------------------------------------
RESEARCH.AIM.COLUMN <- 'Cut_P2'
RESEARCH.DATA.TRAIN <- dt.train
RESEARCH.VARS.TRAIN <- 2:1777 #svi$ix[1:1000]#

trCtrl <- trainControl(method = "repeatedcv", number = 5, repeats = 5, search = "random")


Sys.time()
set.seed(4135)
fitted.model.1 <- train(x = RESEARCH.DATA.TRAIN[, RESEARCH.VARS.TRAIN],
                   y = as.factor(RESEARCH.DATA.TRAIN[, RESEARCH.AIM.COLUMN] > 0),
                   method = "svmRadial",
                   trControl = trCtrl,
                   tuneLength = 6)
Sys.time()
set.seed(4135)
fitted.model.2 <- train(x = RESEARCH.DATA.TRAIN[, RESEARCH.VARS.TRAIN],
                        y = as.factor(RESEARCH.DATA.TRAIN[, RESEARCH.AIM.COLUMN] > 0),
                        method = "xgbTree",
                        trControl = trCtrl,
                        tuneLength = 4)
#--------------------------------------------------------------------
#                            Selecting Featues  
#--------------------------------------------------------------------
trCtrl.down <- trainControl(method = "repeatedcv", number = 5, repeats = 5, search = "random", sampling = "down")

svm.vi <- varImp(fitted.model.1)
svm.svi <- sort(svm.vi$importance$FALSE., decreasing = T, index.return = T)

xgb.vi <- varImp(fitted.model.2)
xgb.svi <- sort(xgb.vi$importance$Overall, decreasing = T, index.return = T)

rf.vi <- varImp(fitted.model.3)
rf.svi <- sort(rf.vi$importance$Overall, decreasing = T, index.return = T)

Sys.time()
set.seed(4135)
fitted.select.1a <- train(x = RESEARCH.DATA.TRAIN[, svm.svi$ix[1:120]],
                         y = as.factor(as.numeric(RESEARCH.DATA.TRAIN[, RESEARCH.AIM.COLUMN])),
                         method = "svmRadial",
                         trControl = trCtrl,
                         tuneLength = 6)
Sys.time()
set.seed(4117)
RESEARCH.DATA.TRAIN.G1 <- subset(RESEARCH.DATA.TRAIN, RESEARCH.DATA.TRAIN$clust == 1)
fitted.select.plus2.2c <- train(x = RESEARCH.DATA.TRAIN[, row.names(xgb.vi$importance)[1:80]],
                         y = as.factor(as.numeric(RESEARCH.DATA.TRAIN[, 'Cut_P2'])),
                         method = "xgbTree",
                         trControl = trCtrl,
                         tuneLength = 3)
Sys.time()
set.seed(4135)
fitted.model.3a <- train(x = RESEARCH.DATA.TRAIN[, svi.rf$ix[1:300]],
                        y = as.factor(RESEARCH.DATA.TRAIN[, RESEARCH.AIM.COLUMN] > 0),
                        method = "rf",
                        trControl = trCtrl,
                        tuneLength = 3)
Sys.time()
#--------------------------------------------------------------------
#                              Testing results
#--------------------------------------------------------------------
testing.data <- dt.test
testing.model <- xgb.2Factor.retP1.g2

median.y.pos <- median(dt.train.g1$Ret_PlusOne[dt.train.g1$Ret_PlusOne > 0], na.rm = T)
median.y.neg <- median(dt.train.g1$Ret_PlusOne[dt.train.g1$Ret_PlusOne < 0], na.rm = T)

zero.result <- getWmae(fitted.model.2,
                       testing.data[, RESEARCH.VARS.TRAIN],
                       testing.data[, RESEARCH.AIM.COLUMN],
                       testing.data$Weight_Daily, rep(0, 2))

const <- 1
test.result.1 <- getWmae(fitted.model.1,
                         testing.data[, RESEARCH.VARS.TRAIN],
                         testing.data[, RESEARCH.AIM.COLUMN],
                         testing.data$Weight_Daily, c(median.y.neg/const, median.y.pos/const))
test.result.2 <- getWmae(fitted.model.2,
                         testing.data[, row.names(xgb.vi$importance)[1:60]],
                         testing.data[, RESEARCH.AIM.COLUMN],
                         testing.data$Weight_Dailfity, c(median.y.neg/const, median.y.pos/const))
test.result.3 <- getWmae(fitted.model.3,
                         testing.data[, RESEARCH.VARS.TRAIN],
                         testing.data[, RESEARCH.AIM.COLUMN],
                         testing.data$Weight_Daily, c(median.y.neg/const, median.y.pos/const))

y <- (test.result.2$y + test.result.3$y)/2
wmae <- checkY(y, testing.data[, RESEARCH.AIM.COLUMN], testing.data$Weight_Daily)
#--------------------------------------------------------------------
#                            END  Testing results
#--------------------------------------------------------------------




#--------------------------------------------------------------------
#                              Getting submission
#--------------------------------------------------------------------

const <- 1
suffix <- "_61"

dt.subm <- read.csv('sample_submission.csv')

dt.subm.g1 <- test.original.data#subset(test.original.data, test.original.data$clust == 1)
dt.subm.g1$Id.61 <- paste(dt.subm.g1$Id, suffix, sep = '')

dt.subm.g1$y.1 <- getY(fitted.select.1a,
                     dt.subm.g1[, svm.svi$ix[1:120]],
                     c(median.y.neg/const, 0,  median.y.pos/const))
dt.subm.g1$y.2 <- getY(fitted.select.2c,
                       dt.subm.g1[, row.names(xgb.vi$importance)[1:80]],
                       c(median.y.neg, 0, median.y.pos))
dt.subm.g1$y.3 <- getY(fitted.model.3,
                       dt.subm.g1[, RESEARCH.VARS.TRAIN],
                       c(median.y.neg/const, median.y.pos/const))

dt.subm$Predicted[dt.subm$Id %in% dt.subm.g1$Id.61] <- (dt.subm.g1$y.2)/15

write.csv(dt.subm, file = 'winton_test_3_try.xgb.top.plus_minus.div15.p03.csv', row.names = F, quote = F)
#--------------------------------------------------------------------
#                            END  Getting submission
#--------------------------------------------------------------------





