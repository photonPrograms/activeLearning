import java.io.*;
import java.util.*;

public class TechniqueExperiment {
    // what fraction of the training data is labeled
    public static final double labelFrac = 0.1;

    // what percent of instance to label from pool
    public static final String[] queryFracs = new String[] {"0.1", "0.2", "0.3", "0.4", "0.5"};

    // training directory
    public static final String trainDir = "./data/mnist_png/training";

    // testing directory
    public static final String testDir = "./data/mnist_png/testing";

    // number of classes
    public static final int numClasses = 10;

    // epochs of training
    public static final int cnnEpochs = 5;
    public static final int commEpochs = 3;

    public static ArrayList<String> runUncertaintyTechnique(
        int trials, int trainingInstances, int testingInstances, String type, int epochs)
        throws IOException {
        /* to run a technique and record results
         * params:
         * : trials: how many times to repeat the experiment
         * : trainingInstances: how many training instances to take labeled + pool
         * : testingInstances: how many testing instances to take
         * : type: the type of uncertainty based learner
         * : epochs: epochs for training
         */

        ArrayList<Image> labeledImages = ImageOps.getImages(
            trainDir, (int) Math.floor(labelFrac * trainingInstances),
            0, numClasses
        );
        ArrayList<Image> poolImages = ImageOps.getImages(
            trainDir, (int) Math.floor(trainingInstances * (1 - labelFrac)), 
            (int) Math.floor(trainingInstances * labelFrac),
            numClasses
        );
        ArrayList<Image> testImages = ImageOps.getImages(
            testDir,
            testingInstances, numClasses
        );
        ArrayList<Image> trainImages;
        ArrayList<Image> totalTrainImages = ImageOps.combineLists(labeledImages, poolImages);
        PoolLearner poolLearner;
        ArrayList<String> results = new ArrayList<>();

        for (int trial = 1; trial <= trials; trial++) {
            CNN cnn = new CNN(
                new Conv3x3(8),
                new MaxPooling(2),
                new Dense(13 * 13 * 8, numClasses)
            );
            cnn.compile(5e-4);
            cnn.train(labeledImages, cnnEpochs);

            if (type == "sm")
                poolLearner = new SmallestMarginLearner(cnn);
            else if (type == "rc")
                poolLearner = new RatioConfidenceLearner(cnn);
            else if (type == "lc")
                poolLearner = new LeastConfidenceLearner(cnn);
            else if (type == "en")
                poolLearner = new EntropyLearner(cnn);
            else {
                System.out.println("No learner matches");
                return null;
            }
            poolLearner.assignScores(poolImages);

            for (String frac: queryFracs) {
                double poolFrac = Double.parseDouble(frac);
                cnn = new CNN(
                    new Conv3x3(8),
                    new MaxPooling(2),
                    new Dense(13 * 13 * 8, numClasses)
                );
                cnn.compile(5e-4);
                trainImages = ImageOps.combineLists(
                    labeledImages,
                    poolLearner.getImages(poolImages, 0.1 / (1 - poolFrac), false)
                );
                String currResult = String.format(
                    "%s, %.3f, %.3f, %.3f",
                    poolFrac,
                    cnn.train(trainImages, epochs).get("Accuracy"),
                    cnn.test(totalTrainImages).get("Accuracy"),
                    cnn.test(testImages).get("Accuracy")
                );
                results.add(currResult);
            }
        }
        return results;
    }

    public static ArrayList<String> runQBCTechnique(
        int trials, int trainingInstances, int testingInstances, String type, int epochs)
        throws IOException {
        /* to run a technique and record results
         * params:
         * : trials: how many times to repeat the experiment
         * : trainingInstances: how many training instances to take labeled + pool
         * : testingInstances: how many testing instances to take
         * : type: the type of uncertainty based learner
         * : epochs: epochs for training
         */

        ArrayList<Image> labeledImages = ImageOps.getImages(
            trainDir, (int) Math.floor(labelFrac * trainingInstances),
            0, numClasses
        );
        ArrayList<Image> poolImages = ImageOps.getImages(
            trainDir, (int) Math.floor(trainingInstances * (1 - labelFrac)), 
            (int) Math.floor(trainingInstances * labelFrac),
            numClasses
        );
        ArrayList<Image> testImages = ImageOps.getImages(
            testDir,
            testingInstances, numClasses
        );
        ArrayList<Image> trainImages;
        ArrayList<Image> totalTrainImages = ImageOps.combineLists(labeledImages, poolImages);
        PoolLearner poolLearner;
        ArrayList<String> results = new ArrayList<>();

        for (int trial = 1; trial <= trials; trial++) {
            Committee committee = new Committee(
                5, 8, 2, 13 * 13 * 8, numClasses
            );
            committee.compile(5e-4);
            committee.train(labeledImages, commEpochs);

            if (type == "ve")
                poolLearner = new VoteEntropyLearner(committee);
            else if (type == "kl")
                poolLearner = new KLDivergenceLearner(committee);
            else {
                System.out.println("No learner matches");
                return null;
            }
            poolLearner.assignScores(poolImages);

            for (String frac: queryFracs) {
                double poolFrac = Double.parseDouble(frac);
                CNN cnn = new CNN(
                    new Conv3x3(8),
                    new MaxPooling(2),
                    new Dense(13 * 13 * 8, numClasses)
                );
                cnn.compile(5e-4);
                trainImages = ImageOps.combineLists(
                    labeledImages,
                    poolLearner.getImages(poolImages, 0.1 / (1 - poolFrac), false)
                );
                String currResult = String.format(
                    "%s, %.3f, %.3f, %.3f",
                    poolFrac,
                    cnn.train(trainImages, epochs).get("Accuracy"),
                    cnn.test(totalTrainImages).get("Accuracy"),
                    cnn.test(testImages).get("Accuracy")
                );
                results.add(currResult);
            }
        }
        return results;
    }

    public static ArrayList<String> runRandomTechnique(
        int trials, int trainingInstances, int testingInstances, String type, int epochs)
        throws IOException {
        /* to run a technique and record results
         * params:
         * : trials: how many times to repeat the experiment
         * : trainingInstances: how many training instances to take labeled + pool
         * : testingInstances: how many testing instances to take
         * : type: the type of uncertainty based learner
         * : epochs: epochs for training
         */

        ArrayList<Image> labeledImages = ImageOps.getImages(
            trainDir, (int) Math.floor(labelFrac * trainingInstances),
            0, numClasses
        );
        ArrayList<Image> poolImages = ImageOps.getImages(
            trainDir, (int) Math.floor(trainingInstances * (1 - labelFrac)), 
            (int) Math.floor(trainingInstances * labelFrac),
            numClasses
        );
        ArrayList<Image> testImages = ImageOps.getImages(
            testDir,
            testingInstances, numClasses
        );
        ArrayList<Image> trainImages;
        ArrayList<Image> totalTrainImages = ImageOps.combineLists(labeledImages, poolImages);
        PoolLearner poolLearner;
        ArrayList<String> results = new ArrayList<>();

        for (int trial = 1; trial <= trials; trial++) {
            poolLearner = new RandomLearner(poolImages.size());
            poolLearner.assignScores(poolImages);

            for (String frac: queryFracs) {
                double poolFrac = Double.parseDouble(frac);
                CNN cnn = new CNN(
                    new Conv3x3(8),
                    new MaxPooling(2),
                    new Dense(13 * 13 * 8, numClasses)
                );
                cnn.compile(5e-4);
                trainImages = ImageOps.combineLists(
                    labeledImages,
                    poolLearner.getImages(poolImages, 0.1 / (1 - poolFrac), false)
                );
                String currResult = String.format(
                    "%s, %.3f, %.3f, %.3f",
                    poolFrac,
                    cnn.train(trainImages, epochs).get("Accuracy"),
                    cnn.test(totalTrainImages).get("Accuracy"),
                    cnn.test(testImages).get("Accuracy")
                );
                results.add(currResult);
            }
        }
        return results;
    }
}
