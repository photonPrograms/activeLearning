import java.util.*;

public class LeastConfidenceLearner extends UncertaintyLearner {
    /* uncertainty sampling with least confidence metric */

    public LeastConfidenceLearner(CNN cnn) {
        super(cnn);
    }

    public void assignScore(Image image) {
        double[] probs = this.cnn.testOneImage(image);
        double minDoubt = Double.MAX_VALUE;
        for (double prob: probs)
            minDoubt = Math.min(minDoubt, 1 - prob);
        image.lcScore = minDoubt;
    }

    public void orderImages(ArrayList<Image> images) {
        Collections.sort(
            images,
            (a, b) -> Double.compare(b.lcScore, a.lcScore) // descending order
        );
    }

    public double getThreshold(ArrayList<Image> images, double fracRequired) {
        ArrayList<Image> compatibleList = getImages(images, fracRequired);
        orderImages(compatibleList);
        return compatibleList.get(compatibleList.size() - 1).lcScore;
    }
}
