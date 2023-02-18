public class Cluster {
    /* info about a cluster */

    public int clusterId; // the uid of the cluster
    public double[][] centroid; // the centroid of the cluster - an image
    public int label; // the label (majority class label)

    public Cluster(int clusterId, double[][] seed) {
        this.clusterId = clusterId;
        centroid = new double[seed.length][seed[0].length];
        for (int i = 0; i < seed.length; i++)
            for (int j = 0; j < seed[0].length; j++)
                centroid[i][j] = seed[i][j];
    }
}