library(RSNNS)
#RSNNS::getSnnsRFunctionTable()   nos da todos los parámetros de snns
set.seed(1)



#CARGA DE LOS DATOS
# los ficheros se deber?an llamar Train1.csv, Test1.csv, Train2.csv, Test2.csv, Train3.csv, Test3.csv
# se asigna a fold 1, 2 o 3

fold <- 3
# formato csv. Campos separados por comas y n?meros con . como separador decimal
trainSet <- read.csv(paste("Train",fold,".txt",sep=""),dec=".",sep=" ",header = F)
testSet  <- read.csv(paste("Test", fold,".txt",sep=""),dec=".",sep=" ",header = F)

#SELECCION DE LA SALIDA. Num de columna del target
nTarget <- ncol(trainSet)

#SEPARAR ENTRADA DE LA SALIDA
trainInput <- trainSet[,-nTarget]
testInput <-  testSet[,-nTarget]

#TRANSFORMAR LA SALIDA DISCRETA A NUMERICA (Matriz con 4 columnas, una por etiqueta, hay un 1 por cada fila en la columna que pertenece a la clase)
trainTarget <- decodeClassLabels(trainSet[,nTarget])
testTarget <-  decodeClassLabels(testSet[,nTarget])

# transformar las entradas de dataframe a matrix para mlp: 
trainInput <- as.matrix(trainInput)
testInput  <- as.matrix(testInput )


#SELECCION DE LOS PARAMETROS
topologia        <- c(5,5)
razonAprendizaje <- 0.05
ciclosMaximos    <- 5000
## asignar nombre de fichero seg?n los par?metros
fileID <- paste("fX",fold,"_topX",paste(topologia,collapse="-"),"_ra",razonAprendizaje,"_CMX",ciclosMaximos,".csv",sep="")


#EJECUCION DEL APRENDIZAJE Y GENERACION DEL MODELO
model <- mlp(x= trainInput,
             y= trainTarget,
             inputsTest= testInput,
             targetsTest= testTarget,
             size= topologia,
             maxit=ciclosMaximos,
             learnFuncParams=c(razonAprendizaje),
             shufflePatterns = F
)

#GRAFICO DE LA EVOLUCION DEL ERROR

plotIterativeError(model)
#fileID


#GENERAR LAS PREDICCIONES en bruto (valores reales)
trainPred <- predict(model,trainInput)
testPred  <- predict(model,testInput)

#CALCULO DE LAS MATRICES DE CONFUSION
trainCm <- confusionMatrix(trainTarget,trainPred)
testCm  <- confusionMatrix(testTarget,testPred)

trainCm
testCm

#PORCENTAJE TOTAL DE ACIERTOS a partir de la matriz de confusi?n
accuracy <- function (cm) sum(diag(cm))/sum(cm)
accuracies <- c(TrainAccuracy= accuracy(trainCm), TestAccuracy=  accuracy(testCm) )
print(accuracies)

#TABLA CON LOS ERRORES POR CICLO
iterativeErrors <- data.frame(MSETrain= (model$IterativeFitError/nrow(trainSet)),
                     MSETest= (model$IterativeTestError/nrow(testSet)))


# calcular errores finales MSE
MSEtrain <-sum((trainTarget - trainPred)^2)/nrow(trainSet)
MSEtest <-sum((testTarget - testPred)^2)/nrow(testSet)


####calcular la CLASE de salida
# transforma las tres columnas reales en la clase 1,2,3,4 segun el maximo de los cuatro valores. 

trainPredClass<-as.factor(apply(trainPred,1,which.max))  
testPredClass<-as.factor(apply(testPred,1,which.max)) 




# #GUARDADO DE RESULTADOS
# #MODELO
# saveRDS(model,             paste("nnet_",gsub("\\.csv","",fileID),".rds",sep=""))
# write.csv(accuracies,     paste("finalAccuracies_",fileID,sep=""))
# write.csv(iterativeErrors,paste("iterativeErrors_",fileID,sep=""))
# #salidas de test en bruto
# write.csv(testPred ,       paste("TestRawOutputs_",fileID,sep=""), row.names = FALSE)
# write.csv(testPredClass,   paste("TestClassOutputs_",fileID,sep=""),row.names = FALSE)
# # matrices de confusi?n
# write.csv(trainCm,        paste("trainCm_",fileID,sep=""))
# write.csv(testCm,         paste("testCm_",fileID,sep=""))
# 
