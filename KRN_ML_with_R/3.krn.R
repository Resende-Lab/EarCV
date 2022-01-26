args = commandArgs(trailingOnly=TRUE)

print(args[1])
print(args[2])
print(args[3])
print(args[4])
print(args[5])

experiment <- as.numeric(args[1])
trait <- as.numeric(args[2])
cv <- as.numeric(args[3])
NR <- as.numeric(args[4])
eta <- as.numeric(args[5])
subsample <- as.numeric(args[6])
lambda <- as.numeric(args[7])
alpha <- as.numeric(args[8])

#experiment <- 1
#trait = 2
#cv = 1

library(caret)
library(dplyr)
library(tidyr)
library(reshape2)
library(xgboost)

# Load data
df = read.csv("./data/hastings_validation.csv")

traits <- colnames(df)[2:3]
ftrs <- colnames(df)[4:ncol(df)]

df[,ftrs] <- scale(df[,ftrs])

mods <- c("BayesA", "RandomForest", "Xgboost")

NMOD <- length(mods)
NPHEN <- length(traits)
NFTRS <- length(ftrs)
NSAMP <- nrow(df)
NFOLD <- 10
NSUB <- NSAMP %% NFOLD
#NR <- 1000

# Gather Initial Seeds
set.seed(0)
seeds <- sample(1:1e4, 100)

seed <- seeds[experiment]
set.seed(seed)

df <- df[sample(1:NSAMP, NSAMP-NSUB),]
idx <- sample(rep(1:NFOLD, (NSAMP-NSUB)/NFOLD))

rownames(df) <- df$Concat
#df$Ground_Fancy<- as.factor(df$Ground_Fancy)

df$Ground_KRN[df$Ground_KRN == 20] <- 18
df$Ground_KRN[df$Ground_KRN == 12] <- 14
df$Ground_KRN <- (df$Ground_KRN / 2) - 7



set.seed(paste(seed, trait, cv, sep = ""))
trnX <- as.matrix(df[, ftrs])
trnY <- data.frame(df[, c(trait + 1, 1)])
trnY[idx == cv, 1] <- NA
tstY <- as.matrix(df[idx == cv, trait + 1])

fit <- xgboost(data = as.matrix(df[idx != cv, ftrs]), 
               label = as.matrix(df[idx != cv, traits[trait]]), 
               nrounds = NR,
               eta = eta,
               subsample = subsample,
	       lambda = lambda,
	       alpha = alpha,
               #                params = params,
               num_class = length(unique(df[, traits[trait]])),
               booster = "gbtree",
               objective="multi:softprob",
               eval_metric = "mlogloss",
               verbose = F)

pred <- predict(fit, as.matrix(df[idx == cv, ftrs]), reshape =T)
pred2 <- max.col(pred) - 1

if(traits[trait] == "Ground_KRN"){
  pred2 <- (pred2 + 7) * 2
  tstY <- (tstY + 7) * 2
}

results <- data.frame(experiment = experiment,
                      trait = traits[trait],
                      cv = cv,
                      nrounds = NR,
                      eta = eta,
                      subsample = subsample,
                      lambda = lambda,
                      alpha = alpha,
		      id = df$Concat[idx == cv],
		      krnPeaks = df$KRN_Pred[idx == cv],
                      y = tstY, 
                      yhat = pred2,
                      pred)

# Create results file
# file <- paste(traits[trait], ".txt", sep = "")  
write.table(results, "results.txt", row.names = F, col.names = F, append = T, quote = F)

#confusionMatrix(factor(results$yhat), reference = factor(results$y))

