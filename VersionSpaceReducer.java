import java.util.*;

public class VersionSpaceReducer {
    /* greedy reduction of version space */

    public Committee committee; // committee defining the version space
    public ArrayList<Image> pool; // the pool of images from which version space is constructed
    public int numClasses; // number of classes in the classification problem

    public VersionSpaceReducer(Committee committee, ArrayList<Image> pool, int numClasses) {
        this.committee = committee;
        this.pool = pool;
        this.numClasses = numClasses;
    }

    public void assignTrueScore(Image image) {
        /* get the version space size obtained after training with the image
         * if we knew the true label of the image
         * params:
         * : image: the image to train with
         */

        Committee comm = this.committee.copy();
        comm.trainOneImage(image);
        image.vsScore = getVersionSpaceSize(comm);
    }

    public void assignWorstScore(Image image) {
        /* get the worst version space score
         * by training with all possible labels for the image
         * params:
         * : image: the image to train with
         */

        int trueLabel = image.label;
        int trueVsScore = image.vsScore;
        int worstVsScore = Integer.MIN_VALUE;
        for (int i = 0; i < numClasses; i++) {
            image.label = i;
            assignTrueScore(image);
            worstVsScore = Math.max(image.vsScore, worstVsScore);
        }
        image.label = trueLabel;
        image.vsScore = trueVsScore;
        image.worstVsScore = worstVsScore;
    }

    public void assignTrueScores(ArrayList<Image> images) {
        /* assign each image its true version space score
         * params:
         * : images: the list of images to be assigned scores
         */

        for (Image image: images) {
            assignTrueScore(image);
            // System.out.print(image.vsScore + " ");
        }
        // System.out.println();
    }

    public void assignWorstScores(ArrayList<Image> images) {
        /* assign each image its worst version space score
         * params:
         * : images: the list of images to be assigned scores
         */

        for (Image image: images) {
            assignWorstScore(image);
            // System.out.print(image.worstVsScore + " ");
        }
        // System.out.println();
    }

    public void order(ArrayList<Image> images, boolean useLabels, boolean assign) {
        /* order the images according to their version space scores
         * params:
         * : images: the images to order
         * : useLabels: whether to use the true scores
         * : assign: if assignment should be done before ordering
         */

        if (useLabels) {
            if (assign)
                assignTrueScores(images);
            Collections.sort(
                images,
                (a, b) -> (a.vsScore - b.vsScore)
            );
        }
        
        else {
            if (assign)
                assignWorstScores(images);
            Collections.sort(
                images,
                (a, b) -> (a.worstVsScore - b.worstVsScore)
            );
        }
    }

    public int getVersionSpaceSize(Committee comm) {
        /* get the current size of version space
         * in terms of points inside it
         * params:
         * : comm: the committee used
         */

        int versionSpaceCount = 0;
        for (Image image: pool) {
            int pred = VectorOps.argmax(comm.classifiers[0].testOneImage(image));
            for (CNN cnn: comm.classifiers) {
                if (VectorOps.argmax(cnn.testOneImage(image)) != pred) {
                    versionSpaceCount++;
                    break;
                }
            }
        }
        return versionSpaceCount;
    }

    public int getVersionSpaceSize() {
        return getVersionSpaceSize(this.committee);
    }

    public void getOverlap(ArrayList<Image> images, boolean useLabels) {
        /* get overlap of the given list with the pool */
        int N = images.size();
        Set<Image> hs = new HashSet<>();
        for (Image image: images)
            hs.add(image);
        order(pool, useLabels, false);
        int overlap = 0;
        for (int i = 0; i < N; i++)
            if (hs.contains(pool.get(i)))
                overlap++;
        System.out.println(
            String.format(
                "%d out of %d: %.3f",
                overlap, N, (double) overlap / N
            )
        );
    }
}