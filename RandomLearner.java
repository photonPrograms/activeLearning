import java.util.*;

public class RandomLearner extends PoolLearner {
    /* for active learning by random selection
     */

    public int N; // the number of images to be handled
    public int[] scores; // random scores for each index
    public int mainIndex; // keeping track of images passed until now

    public RandomLearner(int N) {
        this.N = N;
        scores = new int[N];

        List<Integer> scoreList = new ArrayList<>();
        for (int i = 0; i < N; i++)
            scoreList.add(i);
        Collections.sort(scoreList);

        for (int i = 0; i < N; i++)
            scores[i] = scoreList.get(i);

        mainIndex = 0;
    }

    public void assignScores(ArrayList<Image> images) {
        if (this.N < images.size()) {
            System.out.println("Random assignment failed!");
            return;
        }

        int index = 0;
        for (Image image: images)
            image.randomScore = this.scores[index++];
    }

    public void assignScore(Image image) {
        image.randomScore = this.scores[this.mainIndex];
        this.mainIndex++;
        if (this.mainIndex >= this.N)
            this.mainIndex %= this.N;
    }

    public void orderImages(ArrayList<Image> images) {
        Collections.sort(
            images,
            new Comparator<Image>() {
                public int compare(Image a, Image b) {
                    return Double.compare(a.randomScore, b.randomScore);
                }
            }
        );
    }

    public double getThreshold(ArrayList<Image> images, double fracRequired) {
        return Double.MIN_VALUE;
    }
}
