javac ImageOps.java MatrixOps.java PrettyPrint.java Image.java VectorOps.java
javac Conv3x3.java MaxPooling.java Dense.java CNN.java Committee.java
javac PoolLearner.java UncertaintyLearner.java StreamLearner.java QBCLearner.java
javac EntropyLearner.java RatioConfidenceLearner.java SmallestMarginLearner.java LeastConfidenceLearner.java
javac RandomLearner.java VoteEntropyLearner.java KLDivergenceLearner.java
javac Cluster.java Kmeans.java VersionSpaceReducer.java AggregateResults.java
javac TechniqueExperiment.java Driver.java
java Driver
gnuplot -p plotEpochAcc.p
gnuplot -p plotEpochLoss.p
gnuplot -p plotStepAcc.p
gnuplot -p plotStepLoss.p
python3 learningCurve.py
python3 plotVersionSpaceDistr.py