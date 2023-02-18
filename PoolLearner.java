import java.util.*;

public abstract class PoolLearner {
    /* general framework for pool based learning
     */

    /* assign score to image(s) for ordering later */
    public abstract void assignScores(ArrayList<Image> images);
    public abstract void assignScore(Image image);

    /* order using the assigned scores */
    public abstract void orderImages(ArrayList<Image> images);

    /* get threshold for stream based clustering */
    public abstract double getThreshold(ArrayList<Image> images, double fracRequired);

    /* get a certain number of images after ordering first */
    public ArrayList<Image> getImages(ArrayList<Image> images, int numRequired) {
        if (images.size() < numRequired) {
            System.out.println("Number of images insufficient!");
            return images;
        }
        orderImages(images);
        ArrayList<Image> requiredList = new ArrayList<>();
        for (int i = 0; i < numRequired; i++)
            requiredList.add(images.get(i));
        return requiredList;
    }

    /* get a certain fraction of all images */
    public ArrayList<Image> getImages(ArrayList<Image> images, double fracRequired) {
        if (fracRequired > 1) {
            System.out.println("Number of images insufficient!");
            return images;
        }

        int numRequired = (int) Math.floor(fracRequired * images.size());
        assignScores(images);
        orderImages(images);
        ArrayList<Image> requiredList = new ArrayList<>();
        for (int i = 0; i < numRequired; i++)
            requiredList.add(images.get(i));
        return requiredList;
    }

    public ArrayList<Image> getImages(ArrayList<Image> images, double fracRequired, boolean assign) {
        if (fracRequired > 1) {
            System.out.println("Number of images insufficient!");
            return images;
        }

        int numRequired = (int) Math.floor(fracRequired * images.size());
        if (assign)
            assignScores(images);
        orderImages(images);
        ArrayList<Image> requiredList = new ArrayList<>();
        for (int i = 0; i < numRequired; i++)
            requiredList.add(images.get(i));
        return requiredList;
    }
}