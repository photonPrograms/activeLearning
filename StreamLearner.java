import java.util.*;
public class StreamLearner {
    public PoolLearner scorer; // the active learner to evaluate scores for each instance
    public double threshold; // the threshold based on pool learning
    public String type; // the type of active learning technique used

    public StreamLearner(PoolLearner scorer, String type,
        ArrayList<Image> poolImages, double fracRequired) {
        this.scorer = scorer;
        threshold = scorer.getThreshold(poolImages, fracRequired);
        this.type = type;
    }

    public boolean allow(Image image) {
        /* for any particular image decide whether to allow training with it
         * params:
         * : image: the image under question
         */

        scorer.assignScore(image);

        if (type == "entropy")
            return image.entropyScore >= threshold;
        else if (type == "lc")
            return image.lcScore >= threshold;
        else if (type == "sm")
            return image.smScore <= threshold;
        else if (type == "ratio")
            return image.ratioConfScore >= threshold;
        else if (type == "ve")
            return image.veScore >= threshold;
        else if (type == "kl")
            return image.klScore >= threshold;
        else
            return true;
    }
}
