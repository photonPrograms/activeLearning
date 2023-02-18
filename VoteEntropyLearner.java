import java.util.*;

public class VoteEntropyLearner extends QBCLearner {
    /* vote entropy query by committee */

    public VoteEntropyLearner(Committee committee) {
        super(committee);
    }

    public void assignScore(Image image) {
        double[][] probs = this.committee.testOneImage(image);
        int[] votes = new int[probs[0].length];
        for (int i = 0; i < probs.length; i++) {
            double maxProb = Double.MIN_VALUE;
            int maxClass = 0;
            for (int j = 0; j < probs[0].length; j++)
                if (maxProb < probs[i][j]) {
                    maxProb = probs[i][j];
                    maxClass = j;
                }
            votes[maxClass]++;
        }
        double C = (double) this.committee.committeeSize;
        double sum = 0;
        for (int i = 0; i < votes.length; i++)
            sum += -1 * votes[i] / C * Math.log(Math.abs(votes[i]) / C + 1e-6);
        image.veScore = sum;
    }

    public void orderImages(ArrayList<Image> images) {
        Collections.sort(
            images,
            (a, b) -> (Double.compare(b.veScore, a.veScore)) // descending order
        );
    }

    public double getThreshold(ArrayList<Image> images, double fracRequired) {
        ArrayList<Image> compatibleList = getImages(images, fracRequired);
        orderImages(compatibleList);
        return compatibleList.get(compatibleList.size() - 1).veScore;
    }
}
