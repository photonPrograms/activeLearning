import java.util.*;

public class Committee {
    /* a committee of CNNs */
    CNN[] classifiers;
    int committeeSize, numFilters, poolSize, softmaxInputSize, softmaxOutputSize;
    double learningRate;

    public Committee(
        int committeeSize, int numFilters,
        int poolSize, int softmaxInputSize, int softmaxOutputSize
    ) {
        this.committeeSize = committeeSize;
        this.numFilters = numFilters;
        this.poolSize = poolSize;
        this.softmaxInputSize = softmaxInputSize;
        this.softmaxOutputSize = softmaxOutputSize;
        
        this.classifiers = new CNN[committeeSize];
        for (int i = 0; i < committeeSize; i++)
            classifiers[i] = new CNN(
                new Conv3x3(this.numFilters),
                new MaxPooling(this.poolSize),
                new Dense(this.softmaxInputSize, this.softmaxOutputSize)
            );
    }

    public Committee copy() {
        /* return a copy of the present committee */
        Committee comm = new Committee(
            this.committeeSize, this.numFilters,
            this.poolSize, this.softmaxInputSize, this.softmaxOutputSize
        );
        for (int i = 0; i < this.committeeSize; i++)
            comm.classifiers[i] = this.classifiers[i].copy();
        return comm;
    }

    public void compile(double learningRate) {
        /* setting a new learning rate
         * params:
         * : learningRate: the learning rate to be set
         */
        for (CNN classifier: classifiers)
            classifier.compile(learningRate);
    }
    
    public void compile(double learningRate, double epsilon) {
        /* compiling the committee with new hyperparameters
         * params:
         * : learningRate: the training learning rate
         * : epsilon: the gradient clipper
         */
        for (CNN classifier: classifiers)
            classifier.compile(learningRate, epsilon);
    }

    public void train(ArrayList<Image> images, int epochs) {
        /* training the committee with bagging
         * params:
         * : images: the images to be trained with
         * : epochs: the number of epochs to train for
         */

        Random rand = new Random();
        for (int i = 0; i < committeeSize; i++) {
            ArrayList<Image> currTrainImages = new ArrayList<>();
            for (int j = 0; j < images.size(); j++)
                currTrainImages.add(images.get(rand.nextInt(images.size())));
            classifiers[i].train(currTrainImages, epochs);
        }
    }

    public void trainOneImage(Image image) {
        /* forward and backwards passes with just one image
         * params:
         * : image: the image to be trained with
         */
        for (int i = 0; i < committeeSize; i++)
            classifiers[i].passOneImage(image);
    }

    public double[][] testOneImage(Image image) {
        /* obtain the output of committee for just one image
         * and return the probabilities given by each committee member
         * params:
         * : image: the image to be tested
         */

        double[][] output = new double[committeeSize][];
        for (int i = 0; i < committeeSize; i++)
            output[i] = classifiers[i].testOneImage(image);
        return output;
    }
}
