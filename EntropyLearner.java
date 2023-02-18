import java.util.*;

public class EntropyLearner extends UncertaintyLearner {
    /* entropy based uncertainty sampling */

    public EntropyLearner(CNN cnn) {
        super(cnn);
    }

    public void assignScore(Image image) {
        double[] probs = this.cnn.testOneImage(image);
        double sum = 0;
        for (double prob: probs)
            sum += -1 * prob * Math.log(prob + 1e-6);
        image.entropyScore = sum;
    }

    public void orderImages(ArrayList<Image> images) {
        Collections.sort(
            images,
            (a, b) -> Double.compare(b.entropyScore, a.entropyScore) // descending order
        );
    }

    public double getThreshold(ArrayList<Image> images, double fracRequired) {
        ArrayList<Image> compatibleList = getImages(images, fracRequired);
        orderImages(compatibleList);
        return compatibleList.get(compatibleList.size() - 1).entropyScore;
    }
}
