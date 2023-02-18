import java.util.*;

public class KMeans {
    /* K-Means clustering algorithm */

    public Cluster[] clusters; // the clusters formed
    public int numClusters; // the number of clusters to be formed

    public final int costPerLabel = 100;

    public KMeans(int numClusters, ArrayList<Image> labeledImages) {
        this.numClusters = numClusters;
        clusters = new Cluster[numClusters];

        Map<Integer, Image> seedMap = new HashMap<>();
        for (Image image: labeledImages)
            if (!seedMap.containsKey(image.label))
                seedMap.put(image.label, image);

        for (int i = 0; i < numClusters; i++)
            clusters[i] = new Cluster(i, seedMap.getOrDefault(i, labeledImages.get(0)).matrix);

        int[] clusterCount = new int[numClusters];
        for (Image image: labeledImages) {
            clusters[image.label].centroid = MatrixOps.add(
                clusters[image.label].centroid, image.matrix
            );
            clusterCount[image.label]++;
        }
        for (int i = 0; i < numClusters; i++)
            clusters[i].centroid = MatrixOps.mult(
                clusters[i].centroid, (double) 1 / ++clusterCount[i]
            );
    }

    public void run(ArrayList<Image> images, int epochs) {
        /* cluster the given data with kmeans
         * params:
         * : images: the images to be clustered
         * : epochs: epochs after which to stop
         */

        int nrows = clusters[0].centroid.length, ncols = clusters[0].centroid[0].length;
        for (int epoch = 1; epoch <= epochs; epoch++) {
            double[][][] newCentroids = new double[numClusters][nrows][ncols];
            int[] clusterCounts = new int[numClusters];
            for (Image image: images) {
                image.cluster = getNearestCluster(image);
                int clusterChosen = image.cluster.clusterId;
                newCentroids[clusterChosen] = MatrixOps.add(
                    image.matrix, newCentroids[clusterChosen]
                );
                clusterCounts[clusterChosen]++;
            }
            for (int i = 0; i < numClusters; i++) {
                clusterCounts[i] = Math.max(clusterCounts[i], 1);
                clusters[i].centroid = MatrixOps.mult(
                    newCentroids[i], (double) 1 / clusterCounts[i]
                );
            }
        }
    }

    public Cluster getNearestCluster(Image image) {
        /* get the cluster whose centroid is closest to the image
         * params:
         * : image: the image to be assigned a cluster
         */
        double minDist = Double.MAX_VALUE;
        int clusterChosen = 0;
        for (Cluster cluster: clusters) {
            double currDist = MatrixOps.dist(image.matrix, cluster.centroid);
            if (currDist < minDist) {
                minDist = currDist;
                clusterChosen = cluster.clusterId;
            }
        }
        return clusters[clusterChosen];
    }

    public void labelClusters(ArrayList<Image> images, double sampleFraction, int numClasses) {
        /* assign each cluster the label of its majority
         * based on a sample
         * params:
         * : images: the images that were clustered
         * : sampleFraction: the fraction of images to view label of for each cluster
         */

        ArrayList<Image>[] clusterMembers = new ArrayList[numClusters];
        for (int i = 0; i < numClusters; i++)
            clusterMembers[i] = new ArrayList<>();
        for (Image image: images)
            clusterMembers[image.cluster.clusterId].add(image);
        for (int i = 0; i < numClusters; i++) {
            Collections.shuffle(clusterMembers[i]);
            int imagesSampled = (int) Math.ceil(sampleFraction * clusterMembers[i].size());
            int[] votes = new int[numClasses];
            for (int j = 0; j < imagesSampled && j < clusterMembers[i].size(); j++) {
                Image image = clusterMembers[i].get(j);
                image.cost = costPerLabel;
                votes[image.label]++;
            }
            int maxVotes = Integer.MIN_VALUE, chosenClass = 0;
            for (int j = 0; j < numClasses; j++)
                if (votes[j] > maxVotes) {
                    maxVotes = votes[j];
                    chosenClass = j;
                }
            clusters[i].label = chosenClass;
        }
    }

    public void assignLabels(ArrayList<Image> images) {
        /* assign each image the label of majority class in its cluster
         * params:
         * : images: the images to be labeled
         */
        for (Image image: images)
            if (image.cost == costPerLabel)
                image.clusterLabel = image.label;
            else
                image.clusterLabel = image.cluster.label;
    }

    public void describeBenefits(ArrayList<Image> images) {
        /* print cost, average cost, and accuracy of labeling
         * params:
         * : images: the images to be labeled
         */

        double totalCost = 0, accuracySum = 0;
        for (Image image: images) {
            totalCost += image.cost;
            accuracySum += image.label == image.clusterLabel ? 1 : 0;
        }
        System.out.println(
            String.format(
                "Cluster Based Labeling\nTotal Cost = %.2f, Average Cost = %.2f, Accuracy = %.2f",
                totalCost, totalCost / images.size(), (double) accuracySum / images.size()
            )
        );
    }
}