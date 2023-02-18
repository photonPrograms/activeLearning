import java.util.*;

public class SmallestMarginLearner extends UncertaintyLearner {
    /* uncertainty sampling with smallest margin metric */

    public SmallestMarginLearner(CNN cnn) {
        super(cnn);
    }

    public void assignScore(Image image) {
        double[] probs = this.cnn.testOneImage(image);
        Arrays.sort(probs);
        image.smScore = probs[probs.length - 1] - probs[probs.length - 2];
    }

    public void orderImages(ArrayList<Image> images) {
        Collections.sort(
            images,
            (a, b) -> Double.compare(a.smScore, b.smScore) // ascending order
        );
    }

    public double getThreshold(ArrayList<Image> images, double fracRequired) {
        ArrayList<Image> compatibleList = getImages(images, fracRequired);
        orderImages(compatibleList);
        return compatibleList.get(compatibleList.size() - 1).smScore;
    }
}
