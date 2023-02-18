public class Image {
    /* a data structure to hold the image
     * its matrix and label
     * along with various active learning scores
     */

    public String uid; // a unique identifier for the image
    public double[][] matrix; // the image matrix
    public int label; // the true image class label

    public double randomScore; // randomly chosen points
    public double lcScore; // uncertainty sampling with least confidence
    public double smScore; // uncertainty sampling with smallest margins
    public double ratioConfScore; // uncertainty sampling with ratio of confidence
    public double entropyScore; // uncertainty sampling with entropy
    public double veScore; // qbc sampling with vote entropy
    public double klScore; // qbc sampling with kl divergence

    public Cluster cluster; // the cluster of the image
    public int cost; // cost for labeling
    public int clusterLabel; // the label from clustering

    public int vsScore; // the version space score
    public int worstVsScore; // the worst case version space score
}