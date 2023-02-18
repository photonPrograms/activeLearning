import java.util.*;

public class RatioConfidenceLearner extends UncertaintyLearner {
    /* uncertainty sampling with ratio of confidence */

    public RatioConfidenceLearner(CNN cnn) {
        super(cnn);
    }

    public void assignScore(Image image) {
        double[] probs = this.cnn.testOneImage(image);
        Arrays.sort(probs);
        image.ratioConfScore = probs[probs.length - 2] / probs[probs.length - 1];
    }

    public void orderImages(ArrayList<Image> images) {
        Collections.sort(
            images,
            (a, b) -> Double.compare(b.ratioConfScore, a.ratioConfScore) // descending order
        );
    }

    public double getThreshold(ArrayList<Image> images, double fracRequired) {
        ArrayList<Image> compatibleList = getImages(images, fracRequired);
        orderImages(compatibleList);
        return compatibleList.get(compatibleList.size() - 1).ratioConfScore;
    }
}