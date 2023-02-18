import java.util.*;
import java.io.*;
import java.awt.image.*;
import javax.imageio.*;
import java.math.BigDecimal;;

public class CNN {
    /* encapsulates all the layers of the CNN
     * its training and testing processes
     */

    Conv3x3 convLayer;
    MaxPooling maxpoolLayer;
    Dense softmaxLayer;
    double trainAccuracy, trainLoss, testAccuracy, testLoss;
    double learningRate;
    double epsilon;

    ArrayList<Double> stepLoss, stepAcc, epochLoss, epochAcc;

    static final String trainingDir = "./data/mnist_png/training",
                        testingDir = "./data/mnist_png/testing";

    public CNN(Conv3x3 convLayer, MaxPooling maxpoolLayer, Dense softmaxLayer) {
        this.convLayer = convLayer;
        this.maxpoolLayer = maxpoolLayer;
        this.softmaxLayer = softmaxLayer;

        learningRate = 5e-3;
        epsilon = 1e-6;
        softmaxLayer.setLearningRate(this.learningRate);

        trainAccuracy = 0;
        trainLoss = 0;
        testAccuracy = 0;
        testLoss = 0;

        stepLoss = new ArrayList<>();
        stepAcc = new ArrayList<>();
        epochLoss = new ArrayList<>();
        epochAcc = new ArrayList<>();
    }

    public CNN copy() {
        CNN cnn = new CNN(
            new Conv3x3(this.convLayer.numFilters),
            new MaxPooling(this.maxpoolLayer.poolSize),
            new Dense(this.softmaxLayer.inputSize, this.softmaxLayer.outputSize)
        );
        cnn.compile(this.learningRate);
        
        for (int i = 0; i < this.convLayer.filters.length; i++)
            for (int j = 0; j < this.convLayer.filters[0].length; j++)
                for (int k = 0; k < this.convLayer.filters[0][0].length; k++)
                    cnn.convLayer.filters[i][j][k] = this.convLayer.filters[i][j][k];
        
        for (int i = 0; i < this.softmaxLayer.W.length; i++)
            for (int j = 0; j < this.softmaxLayer.W[0].length; j++)
                cnn.softmaxLayer.W[i][j] = this.softmaxLayer.W[i][j];
        
        for (int i = 0; i < this.softmaxLayer.b.length; i++)
            for (int j = 0; j < this.softmaxLayer.b[0].length; j++)
                cnn.softmaxLayer.b[i][j] = this.softmaxLayer.b[i][j];
        return cnn;
    }

    public HashMap<String, Double> passOneImage(int digit, int imageIndex) throws IOException {
        /* forward and backward passes for a single training image
         * params:
         * : digit: the digit for the image (label)
         * : imageIndex: the number of the image within the digit directory
         */

        String fileName = String.format(
            "%s/%d/%d.png", trainingDir, digit, imageIndex
        );

        // forward propagation
        double[][] imageMatrix = ImageOps.getImageMatrix(fileName);
        double[][][] outFromConv = convLayer.forwardProp(imageMatrix);
        double[][][] outFromPool = maxpoolLayer.forwardProp(outFromConv);
        double[][] output = softmaxLayer.forwardProp(outFromPool);

        // is the prediction correct?
        int isCorrect = MatrixOps.argmax(output)[1] == digit ? 1 : 0;

        // categorical crossentropy loss
        double loss = -Math.log(Math.abs(output[0][digit]) + epsilon);

        HashMap<String, Double> individualStats = new HashMap<>();
        individualStats.put("IsCorrect", (double) isCorrect);
        individualStats.put("Loss", loss);

        // backprop
        double[][] gradOut = MatrixOps.zeros(output.length, output[0].length);
        gradOut[0][digit] = -1 / output[0][digit];

        double[][][] gradSoftmax = softmaxLayer.backwardProp(gradOut);
        double[][][] gradPool = maxpoolLayer.backwardProp(gradSoftmax);
        convLayer.backwardProp(gradPool);

        return individualStats;
    }

    public HashMap<String, Double> passOneImage(Image image) {
        /* forward and backward passes for a single training image
         * params:
         * : image: the image to be trained with
         */
        double[][] imageMatrix = image.matrix;
        double[][][] outFromConv = convLayer.forwardProp(imageMatrix);
        double[][][] outFromPool = maxpoolLayer.forwardProp(outFromConv);
        double[][] output = softmaxLayer.forwardProp(outFromPool);

        // is the prediction correct?
        int isCorrect = MatrixOps.argmax(output)[1] == image.label ? 1 : 0;

        // categorical crossentropy loss
        double loss = -Math.log(Math.abs(output[0][image.label]) + epsilon);

        HashMap<String, Double> individualStats = new HashMap<>();
        individualStats.put("IsCorrect", (double) isCorrect);
        individualStats.put("Loss", loss);

        // backprop
        double[][] gradOut = MatrixOps.zeros(output.length, output[0].length);
        gradOut[0][image.label] = -1 / output[0][image.label];

        double[][][] gradSoftmax = softmaxLayer.backwardProp(gradOut);
        double[][][] gradPool = maxpoolLayer.backwardProp(gradSoftmax);
        convLayer.backwardProp(gradPool);

        return individualStats;
    }

    public HashMap<String, String> testOneImage(int digit, int imageIndex) throws IOException {
        /* forward pass and loss calculation for testing
         * params:
         * : digit: the digit for the image (true label)
         * : imageIndex: the number of the image within the digit directory
         */
        String fileName = String.format(
            "%s/%d/%d.png", testingDir, digit, imageIndex
        );
        
        // forward propagation
        double[][] imageMatrix = ImageOps.getImageMatrix(fileName);
        double[][][] outFromConv = convLayer.forwardProp(imageMatrix);
        double[][][] outFromPool = maxpoolLayer.forwardProp(outFromConv);
        double[][] output = softmaxLayer.forwardProp(outFromPool);

        // stats calculation
        int predictedLabel = MatrixOps.argmax(output)[1];
        double loss = -Math.log(Math.abs(output[0][digit]) + epsilon);

        HashMap<String, String> individualStats = new HashMap<>();
        individualStats.put("Predicted", String.valueOf(predictedLabel));
        individualStats.put("Loss", String.valueOf(loss));

        return individualStats;
    }

    public double[] testOneImage(Image image) {
        /* forward pass for one testing image
         * params:
         * : image: the image to be tested upon
         */
        
        // forward propagation
        double[][] imageMatrix = image.matrix;
        double[][][] outFromConv = convLayer.forwardProp(imageMatrix);
        double[][][] outFromPool = maxpoolLayer.forwardProp(outFromConv);
        double[][] output = softmaxLayer.forwardProp(outFromPool);

        double[] probs = new double[output[0].length];
        for (int i = 0; i < probs.length; i++)
            probs[i] = Math.abs(output[0][i]);
        return probs;
    }

    public void compile(double learningRate, double epsilon) {
        this.learningRate = learningRate;
        this.epsilon = epsilon;
        softmaxLayer.setLearningRate(this.learningRate);
        convLayer.setLearningRate(this.learningRate);
    }

    public void compile(double learningRate) {
        this.learningRate = learningRate;
        softmaxLayer.setLearningRate(this.learningRate);
        convLayer.setLearningRate(this.learningRate);
    }

    public void saveSeries() {
        PrettyPrint.saveSeries(epochAcc, epochLoss, stepAcc, stepLoss);
    }

    public void train(int epochs, int imagesPerEpoch) throws IOException {
        /* train the convolutional neural network
         * params:
         * : epochs: the number of epochs for training
         * : imagesPerEpoch: the number of images to be used from training dataset
         */

        int imagesPerDigit = (int) Math.ceil((double) imagesPerEpoch / 10);
        List<Integer> imageIndices = new ArrayList<>();
        for (int i = 1; i <= imagesPerDigit; i++)
            imageIndices.add(i);

        double accuracySum = 0, loss = 0, lossSum = 0, accuracy = 0;
        List<Long> timesTaken = new ArrayList<>(); 
        for (int epoch = 1; epoch <= epochs; epoch++) {
            long startTime = System.currentTimeMillis();
            System.out.println(String.format("--Epoch %d--", epoch));
            accuracySum = 0;
            lossSum = 0;
            loss = 0;
            accuracy = 0;
            int steps = 0;
            Collections.shuffle(imageIndices);

            for (Integer imageIndex: imageIndices) {
                for (int i = 0; i < 10; i++) {
                    Map<String, Double> individualStats = 
                        passOneImage(i, imageIndex.intValue());

                    steps++;
                    accuracySum += individualStats.get("IsCorrect");
                    lossSum += individualStats.get("Loss");

                    if (steps % 100 == 0) {
                        accuracy = accuracySum / steps;
                        loss = lossSum / steps;
                        stepAcc.add(accuracy);
                        stepLoss.add(loss);
                        System.out.println(
                            String.format(
                                "[Step %d]: Accuracy = %.3f, Loss = %.3f",
                                steps, accuracy, loss
                            )
                        );
                    }
                }
            }

            timesTaken.add(System.currentTimeMillis() - startTime);
            accuracy = accuracySum / steps;
            loss = lossSum / steps;
            epochAcc.add(accuracy);
            epochLoss.add(loss);

            System.out.println("\n");
        }

        trainAccuracy = accuracy;
        trainLoss = loss;

        long totalTime = 0;
        for (int i = 0; i < timesTaken.size(); i++)
            totalTime += timesTaken.get(i) / 1000;

        System.out.println(
            String.format(
                "---###---Training Summary---###---\nAccuracy = %.3f\nLoss = %.3f\n",
                trainAccuracy, trainLoss
            )
        );
        System.out.println("Average Time Per Epoch (s) = " + (totalTime / epochs));
        System.out.println(
            "Average Time Per Image (ms) = " + (1000 * (totalTime / epochs) / imagesPerEpoch)
        );
        System.out.println("\n\n");
    }

    public void test(int imagesToTest) throws IOException {
        /* testing phase
         * params:
         * : imagesToTest: the number of images to be considered for testing
         */

        int imagesPerDigit = (int) Math.ceil((double) imagesToTest / 10);
        int[][] confusionMatrix = new int[10][10];

        double accuracySum = 0, loss = 0, lossSum = 0, accuracy = 0;
        
        long startTime = System.currentTimeMillis();
        for (int i = 1; i <= imagesPerDigit; i++)
            for (int j = 0; j < 10; j++) {
                Map<String, String> individualStats = testOneImage(j, i);
                int predLabel = Integer.parseInt(individualStats.get("Predicted"));
                accuracySum += predLabel == j ? 1 : 0;
                lossSum += Double.parseDouble(individualStats.get("Loss"));
                confusionMatrix[j][predLabel]++;
            }
        
        long totalTime = System.currentTimeMillis() - startTime;
        accuracy = accuracySum / (imagesPerDigit * 10);
        loss = lossSum / (imagesPerDigit * 10);
        
        testAccuracy = accuracy;
        testLoss = loss;

        System.out.println(
            String.format(
                "---###---Testing Summary---###---\nAccuracy = %.3f\nLoss = %.3f\n",
                testAccuracy, testLoss
            )
        );
        System.out.println("Average Time Per Image (ms) = " + (totalTime / imagesToTest));
        System.out.println();
        PrettyPrint.displayConfusionMatrix(confusionMatrix);
        System.out.println("\n\n");
    }

    public HashMap<String, Double> train(ArrayList<Image> images, int epochs) {
        double accuracySum = 0, loss = 0, lossSum = 0, accuracy = 0;
        for (int epoch = 1; epoch <= epochs; epoch++) {
            accuracySum = 0;
            lossSum = 0;
            loss = 0;
            accuracy = 0;
            int steps = 0;

            Collections.shuffle(images);
            for (Image image: images) {
                Map<String, Double> individualStats = passOneImage(image);
                steps++;
                accuracySum += individualStats.get("IsCorrect");
                lossSum += individualStats.get("Loss");
            }

            accuracy = accuracySum / steps;
            loss = lossSum / steps;
        }

        trainAccuracy = accuracy;
        trainLoss = loss;
        HashMap<String, Double> trainStats = new HashMap<>();
        trainStats.put("Accuracy", accuracy);
        trainStats.put("Loss", loss);
        return trainStats;
    }

    public HashMap<String, Double> test(ArrayList<Image> images) {
        double accuracySum = 0, accuracy = 0, lossSum = 0, loss = 0;
        for (Image image: images) {
            double[] output = testOneImage(image);
            int predictedLabel = VectorOps.argmax(output);
            lossSum += -Math.log(Math.abs(output[image.label]) + epsilon);
            accuracySum += predictedLabel == image.label ? 1 : 0;
        }
        accuracy = accuracySum / images.size();
        loss = lossSum / images.size();
        testAccuracy = accuracy;
        testLoss = loss;
        HashMap<String, Double> testStats = new HashMap<>();
        testStats.put("Accuracy", accuracy);
        testStats.put("Loss", loss);
        return testStats;
    }
}
