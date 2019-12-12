require(reticulate)
require(SPOT)

use_python("/usr/bin/python3")
#virtualenv_install("r-reticulate", "scipy")
#virtualenv_install("r-reticulate", "deap")
#virtualenv_install("r-reticulate", "matplotlib")
# virtualenv_install("r-reticulate", "multiprocessing")

scipy <- import("scipy")
deap <- import("deap")
timeit <- import("timeit")
numpy <- import("numpy")
operator <- import("operator")
random <- import("random")
functools <- import("functools")
datetime <- import("datetime")

source_python("GP_Goddard_forR.py")
#("testR.py")


library(SPOT)

spotData <- NULL

lower = c(50,  30,  1.2,  0.3, 0.3,  3,  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
#lower = c(100,  20,  1,   1.4,   0.6,  0.6, 4, 1.4, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1)
upper = c(200, 60, 1.6, 0.8, 0.7, 50,  2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
j=1
for(i  in 1:20)
  if(length(res[[i]])==2){
    Res[[j]] = res[[i]]
    j = j+1
  }


resy = map(Res,2) %>% unlist()
resx = map(Res,1)
resx=matrix(unlist(resx),16,,byrow = T)
resx = resx[,-6]
objFun <- function(x){
  
  interm<-function(x){
    
    if(sum(x[9:19])<4)
      return(1e10)
    x = c(x[1:4] , 1-x[4]-0.05,x[5:20])
    #test(x)
    # browser()
    # X = rep(x,2)
    # done = which(apply(resx,1, function(resx,x){
    #  browser; identical(resx,x)},x))
    # 
    # 
    # if (any (as.logical(done)))
    #   return(resy[[done]])
    # 
    
    y=GP_param_tuning(x)
    return(y)
  }
  y = matrix(NA, nrow = nrow(x),1)
  for ( i in 1: nrow(x)){
    print(i)
    y[i]=interm(x[i,,drop=FALSE])
  }
  # as.matrix( apply(x, 1, interm))
  y=as.matrix(y)
  ynew = (y)
  xnew = matrix(x,1,)
  
  if(file.exists("xy.RData")){
    load("xy.RData")
    x = rbind(x,xnew)
    y = rbind(y,ynew)
  }
  save(x, y, file = "xy.RData")
  ynew
}


#objFun(matrix(lower,1))
#   objFun <- SPOT::funSphere
types= c(rep("integer", 2), rep("numeric",3), "integer", "numeric",rep("factor",(length(upper)-7)))
#SPOT Call


# des = designLHD(,lower,upper, control = list(size= 20,types=types) )
# 
# res = objFun(des)

spotData <- spot(
  # x=resx,
  fun = objFun, lower = lower,
  upper = upper,
  control = list(model = buildKriging,
                 modelControl = list(thetaLower = 1e-04,
                                     thetaUpper = 100,
                                     algTheta = optimDE,
                                     budgetAlgTheta = 1000,
                                     optimizeP = TRUE,
                                     useLambda = TRUE,
                                     lambdaLower = -6,
                                     lambdaUpper = 0,
                                     startTheta = NULL,
                                     reinterpolate = FALSE,
                                     types = types,
                                     target="ei"),
                 optimizer = optimDE,
                 optimizerControl = list(funEvals = 30000,
                                         populationSize = 260,
                                         types = types),
                 design = designLHD,
                 designControl = list(size = 20,
                                      retries = 1,
                                      inequalityConstraint = NULL,
                                      replicates = 1,
                                      types = types),
                 funEvals = 30,
                 noise = FALSE,
                 OCBA = FALSE,
                 OCBAbudget = 0,
                 replicates = 1,
                 seedFun = NA,
                 seedSPOT = 1,
                 plots = TRUE,
                 types = types))

