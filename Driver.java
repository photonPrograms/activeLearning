import java.io.IOException;
import java.util.*;

public class Driver {
    public static void testVersionSpace() throws IOException {
        /*
         *********************
         **  VERSION SPACE  ** 
         *********************
         */
        int N = 2500;
        double labelFraction = 0.1;
        ArrayList<Image> labeledImages = ImageOps.getImages(
            "./data/mnist_png/training", 
            (int) Math.floor(N * labelFraction), 0, 10
        );
        ArrayList<Image> poolImages = ImageOps.getImages(
            "./data/mnist_png/training",
            (int) Math.floor(N * (1 - labelFraction)), (int) Math.floor(N * labelFraction),
            10
        );
        ArrayList<Image> testImages = ImageOps.getImages(
            "./data/mnist_png/testing",
            500, 10
        );
        System.out.println(labeledImages.size() + " " + poolImages.size());

        System.out.println("Training committee...");
        Committee committee = new Committee(
            5, 8, 2, 13 * 13 *8, 10
        );
        committee.compile(5e-4);
        committee.train(labeledImages, 3);

        System.out.print("Total Version Space size: ");
        VersionSpaceReducer versionSpaceReducer = new VersionSpaceReducer(
            committee, poolImages, 10
        );
        int vsSize = versionSpaceReducer.getVersionSpaceSize();
        System.out.println(vsSize);

        poolImages = ImageOps.getImages(
            "./data/mnist_png/training",
            (int) Math.floor(N * labelFraction), (int) Math.floor(N * labelFraction),
            10
        );
        System.out.print("New total Version Space size: ");
        versionSpaceReducer = new VersionSpaceReducer(
            committee, poolImages, 10
        );
        vsSize = versionSpaceReducer.getVersionSpaceSize();
        System.out.println(vsSize);
        System.out.println("Assign true scores to " + poolImages.size() + " images: ");
        for (Image image: poolImages) {
            versionSpaceReducer.assignTrueScore(image);
            System.out.println(image.uid + ": " + image.vsScore);
        }
        versionSpaceReducer.order(poolImages, true, false);
        System.out.println("Ordered by true scores:");
        for (Image image: poolImages)
            System.out.println(image.uid + ": " + image.vsScore);
        
        System.out.println("Assign worst scores to " + poolImages.size() + "images: ");
        for (Image image: poolImages) {
            versionSpaceReducer.assignWorstScore(image);
            System.out.println(image.uid + ": " + image.worstVsScore);
        }
        versionSpaceReducer.order(poolImages, false, false);
        // System.out.println("Ordered by worst scores");
        // for (Image image: poolImages)
        //     System.out.println(image.uid + ": " + image.worstVsScore);
    }

    public static void testClustering() throws IOException {
        /*
         ******************
         **  CLUSTERING  ** 
         ******************
         */
        int N = 2500;
        double labelFraction = 0.1;
        ArrayList<Image> labeledImages = ImageOps.getImages(
            "./data/mnist_png/training", 
            (int) Math.floor(N * labelFraction), 0, 10
        );
        ArrayList<Image> poolImages = ImageOps.getImages(
            "./data/mnist_png/training",
            (int) Math.floor(N * (1 - labelFraction) * 0.4), (int) Math.floor(N * labelFraction),
            10
        );
        ArrayList<Image> testImages = ImageOps.getImages(
            "./data/mnist_png/testing",
            500, 10
        );
        System.out.println(labeledImages.size() + " " + poolImages.size());

        // kmeans
        System.out.println("\n--CLUSTERING--\n");
        KMeans kmeans = new KMeans(10, labeledImages);
        kmeans.run(poolImages, 100);
        kmeans.labelClusters(poolImages, 0.2, 10);
        kmeans.assignLabels(poolImages);
        kmeans.describeBenefits(poolImages);
        for (int i = 0; i < kmeans.numClusters; i++)
            System.out.println(
                String.format(
                    "Cluster: %3d, Label: %3d",
                    kmeans.clusters[i].clusterId, kmeans.clusters[i].label
                )
            );

        // cnn results without active learning (only labeled examples)
        System.out.println("\n--Unassisted CNN--\n");
        CNN cnn = new CNN(
            new Conv3x3(8),
            new MaxPooling(2),
            new Dense(13 * 13 * 8, 10)
        );
        cnn.compile(5e-4);
        ArrayList<Image> trainImages = labeledImages;
        HashMap<String, Double> trainStats = cnn.train(trainImages, 5);
        trainImages = ImageOps.combineLists(
            labeledImages,
            poolImages
        );
        HashMap<String, Double> totalTrainStats = cnn.test(trainImages);
        HashMap<String, Double> testStats = cnn.test(testImages);
        System.out.println(
            String.format(
                "Train: %.3f; Total Train: %.3f; Test: %.3f",
                trainStats.get("Accuracy"), totalTrainStats.get("Accuracy"),
                testStats.get("Accuracy")
            )
        );

        System.out.println("\n--Labeling with Clustering--\n");
        for (Image image: poolImages) {
            int temp = image.label;
            image.label = image.clusterLabel;
            image.clusterLabel = temp;
        }
        trainImages = ImageOps.combineLists(
            labeledImages,
            poolImages
        );
        cnn = new CNN(
            new Conv3x3(8),
            new MaxPooling(2),
            new Dense(13 * 13 * 8, 10)
        );
        cnn.compile(5e-4);
        trainStats = cnn.train(trainImages, 5);
        for (Image image: poolImages) {
            int temp = image.label;
            image.label = image.clusterLabel;
            image.clusterLabel = temp;
        }
        trainImages = ImageOps.combineLists(
            labeledImages,
            poolImages
        );
        totalTrainStats = cnn.test(trainImages);
        testStats = cnn.test(testImages);
        System.out.println(
            String.format(
                "Train: %.3f; Total Train: %.3f; Test: %.3f",
                trainStats.get("Accuracy"), totalTrainStats.get("Accuracy"),
                testStats.get("Accuracy")
            )
        );
    }

    public static void testPoolBased(Scanner scan) throws IOException {
        /*
         ***************************
         **  POOL BASED LEARNING  ** 
         ***************************
         */

        System.out.println("--Pool Based Learning--");
        System.out.println("Enter your choice:");
        System.out.println("1 - Random Selection");
        System.out.println("2 - Least Confidence Based Uncertainty Sampling");
        System.out.println("3 - Smallest Margin Based Uncertainty Sampling");
        System.out.println("4 - Entropy Based Uncertainty Sampling");
        System.out.println("5 - Ratio of Confidence Based Uncertainty Sampling");
        System.out.println("6 - Vote Entropy Based QBC");
        System.out.println("7 - KL Divergence Based QBC");

        int choice = scan.nextInt();
        ArrayList<String> results;
        switch (choice) {
            case 1:
                // random selection
                System.out.println("Chosen: Random Selection");
                results = TechniqueExperiment.runRandomTechnique(
                    3, 2500, 500, "rn", 3
                );
                PrettyPrint.writeRawTrials(results, "rn");
                results = PrettyPrint.openRawTrials("rn");
                results = AggregateResults.aggregateResults(results);
                PrettyPrint.writeAggregateTrials(results, "rn");
                break;

            case 2:
                // least confidence
                System.out.println("Chosen: Least Confidence");
                results = TechniqueExperiment.runUncertaintyTechnique(
                    3, 2500, 500, "lc", 3
                );
                PrettyPrint.writeRawTrials(results, "lc");
                results = PrettyPrint.openRawTrials("lc");
                results = AggregateResults.aggregateResults(results);
                PrettyPrint.writeAggregateTrials(results, "lc");
                break;
            
            case 3:
                // smallest margin
                System.out.println("Chosen: Smallest Margin");
                results = TechniqueExperiment.runUncertaintyTechnique(
                    3, 2500, 500, "sm", 3
                );
                PrettyPrint.writeRawTrials(results, "sm");
                results = PrettyPrint.openRawTrials("sm");
                results = AggregateResults.aggregateResults(results);
                PrettyPrint.writeAggregateTrials(results, "sm");
                break;

            case 4:
                // entropy
                System.out.println("Chosen: Entropy");
                results = TechniqueExperiment.runUncertaintyTechnique(
                    3, 2500, 500, "en", 3
                );
                PrettyPrint.writeRawTrials(results, "en");
                results = PrettyPrint.openRawTrials("en");
                results = AggregateResults.aggregateResults(results);
                PrettyPrint.writeAggregateTrials(results, "en");
                break;

            case 5:
                // ratio of confidence
                System.out.println("Chosen: Ratio of Confidence");
                results = TechniqueExperiment.runUncertaintyTechnique(
                    3, 2500, 500, "rc", 3
                );
                PrettyPrint.writeRawTrials(results, "rc");
                results = PrettyPrint.openRawTrials("rc");
                results = AggregateResults.aggregateResults(results);
                PrettyPrint.writeAggregateTrials(results, "rc");
                break;

            case 6:
                // vote entropy
                System.out.println("Chosen: Vote Entropy");
                results = TechniqueExperiment.runQBCTechnique(
                    3, 2500, 500, "ve", 3
                );
                PrettyPrint.writeRawTrials(results, "ve");
                results = PrettyPrint.openRawTrials("ve");
                results = AggregateResults.aggregateResults(results);
                PrettyPrint.writeAggregateTrials(results, "ve");
                break;

            case 7:
                // kl divergence
                System.out.println("Chosen: KL Divergence");
                results = TechniqueExperiment.runQBCTechnique(
                    3, 2500, 500, "kl", 3
                );
                PrettyPrint.writeRawTrials(results, "kl");
                results = PrettyPrint.openRawTrials("kl");
                results = AggregateResults.aggregateResults(results);
                PrettyPrint.writeAggregateTrials(results, "kl");
                break;

            default:
                System.out.println("Invalid Choice!");
        }
    }

    public static void testStreamBased() throws IOException {
        /*
         *****************************
         **  STREAM BASED LEARNING  ** 
         *****************************
         */

        System.out.println("Stream based learning with entropy learner");
        int N = 2500;
        double labelFraction = 0.1;
        ArrayList<Image> labeledImages = ImageOps.getImages(
            "./data/mnist_png/training", 
            (int) Math.floor(N * labelFraction), 0, 10
        );
        ArrayList<Image> poolImages = ImageOps.getImages(
            "./data/mnist_png/training",
            (int) Math.floor(N * (1 - labelFraction)), (int) Math.floor(N * labelFraction),
            10
        );
        ArrayList<Image> testImages = ImageOps.getImages(
            "./data/mnist_png/testing",
            500, 10
        );
        System.out.println(labeledImages.size() + " " + poolImages.size());

        CNN cnn = new CNN(
            new Conv3x3(8),
            new MaxPooling(2),
            new Dense(13 * 13 * 8, 10)
        );
        cnn.compile(5e-4);
        ArrayList<Image> trainImages = labeledImages;

        HashMap<String, Double> trainStats = cnn.train(trainImages, 5);
        trainImages = ImageOps.combineLists(
            labeledImages,
            poolImages
        );
        HashMap<String, Double> totalTrainStats = cnn.test(trainImages);
        HashMap<String, Double> testStats = cnn.test(testImages);
        System.out.println("Unassisted CNN:");
        System.out.println(
            String.format(
                "Train: %.3f; Total Train: %.3f; Test: %.3f",
                trainStats.get("Accuracy"), totalTrainStats.get("Accuracy"),
                testStats.get("Accuracy")
            )
        );
        PoolLearner poolLearner = new EntropyLearner(cnn);
        StreamLearner streamLearner = new StreamLearner(
            poolLearner, "entropy", poolImages, 0.2
        );
        poolImages = ImageOps.getImages(
            "./data/mnist_png/training",
            (int) Math.floor(N * (1 - labelFraction)), 5000,
            10
        );
        System.out.println("Opening the stream!");
        int accepted = 0;
        Collections.shuffle(poolImages);
        for (Image image: poolImages) {
            if (streamLearner.allow(image)) {
                System.out.print(image.uid + ": " + image.entropyScore + " Accepted || ");
                cnn.passOneImage(image);
                accepted++;
            }
            else
                System.out.print(image.uid + ": " + image.entropyScore + " Rejected || ");
        }
        System.out.println("\n" + accepted + " of " + poolImages.size());
        testStats = cnn.test(testImages);
        totalTrainStats = cnn.test(trainImages);
        System.out.println("CNN after stream learning: ");
        System.out.println(
            String.format(
                "Total Train: %.3f; Test: %.3f",
                totalTrainStats.get("Accuracy"),
                testStats.get("Accuracy")
            )
        );
    }

    public static void main(String[] args) throws IOException {
        Scanner scan = new Scanner(System.in);

        /*
         ************
         **  MENU  ** 
         ************
         */
        System.out.println("[[[[[ ASSIGNMENT-2 ]]]]]");
        System.out.println("Enter your choice:");
        System.out.println("1 - Pool based learning");
        System.out.println("2 - Stream based learning");
        System.out.println("3 - Version Space");
        System.out.println("4 - Clustering");

        int choice = scan.nextInt();

        switch (choice) {
            case 1:
                /*
                 ***************************
                 **  POOL BASED LEARNING  ** 
                 ***************************
                 */
                testPoolBased(scan);
                break;
            
            case 2:
                /*
                 *****************************
                 **  STREAM BASED LEARNING  ** 
                 *****************************
                 */
                testStreamBased();
                break;

            case 3:
                /*
                 *********************
                 **  VERSION SPACE  ** 
                 *********************
                 */
                testVersionSpace();
                break;

            case 4:
                /*
                 ******************
                 **  CLUSTERING  ** 
                 ******************
                 */
                testClustering();
                break;

            default:
                System.out.println("Invalid choice.");
        }
        scan.close();
    }
}
