import java.util.*;

public abstract class QBCLearner extends PoolLearner {
    /* framework for query by committee learning */

    public Committee committee; // committee of classifiers

    public QBCLearner(Committee committee) {
        this.committee = committee;
    }

    public void assignScores(ArrayList<Image> images) {
        for (Image image: images)
            assignScore(image);
    }
}
