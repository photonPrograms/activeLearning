import java.util.*;

public class KLDivergenceLearner extends QBCLearner {
    /* query by committee learning with KL divergence criterion */

    public KLDivergenceLearner(Committee committee) {
        super(committee);
    }

    public void assignScore(Image image) {
        double[][] probs = this.committee.testOneImage(image);
        double[] consensus = new double[probs[0].length];
        for (int i = 0; i < probs[0].length; i++) {
            double consensusSum = 0;
            for (int j = 0; j < probs.length; j++)
                consensusSum += Math.abs(probs[j][i]);
            consensus[i] = Math.abs(consensusSum / probs.length);
        }
        double[] D = new double[probs.length];
        for (int i = 0; i < probs.length; i++) {
            D[i] = 0;
            for (int j = 0; j < probs[0].length; j++)
                D[i] += probs[i][j] * Math.log(Math.abs(probs[i][j] / consensus[j]) + 1e-6);
        }
        double sum = 0;
        for (int i = 0; i < probs.length; i++)
            sum += D[i];
        image.klScore = sum / this.committee.committeeSize;
    }

    public void orderImages(ArrayList<Image> images) {
        Collections.sort(
            images,
            (a, b) -> (Double.compare(b.klScore, a.klScore)) // descending order
        );
    }

    public double getThreshold(ArrayList<Image> images, double fracRequired) {
        ArrayList<Image> compatibleList = getImages(images, fracRequired);
        orderImages(compatibleList);
        return compatibleList.get(compatibleList.size() - 1).klScore;
    }
}
