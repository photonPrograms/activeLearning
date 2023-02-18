import java.util.*;

public abstract class UncertaintyLearner extends PoolLearner {
    /* learning with uncertainty sampling */
    public CNN cnn; // cnn to be used for score assignments

    public UncertaintyLearner(CNN cnn) {
        this.cnn = cnn.copy();
    }

    public void assignScores(ArrayList<Image> images) {
        for (Image image: images)
            assignScore(image);
    }
}