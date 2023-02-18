public class MaxPooling {
    /* max pooling layer 
     */

    // square matrix size to consider for pooling
    public int poolSize;
    public double[][][] cacheInput, cacheOutput;

    public static final double doubleZero = 1e-8;

    public MaxPooling() {
        poolSize = 2;
        cacheInput = null;
        cacheOutput = null;
    }

    public MaxPooling(int poolSize) {
        this.poolSize = poolSize;    
    }

    public double[][] pool(double[][] input, int nrows, int ncols) {
        /* max pool over the image
         * : params
         * : input: the input image 2d matrix to pool over
         * : nrows: the number of rows in resultant matrix
         * : ncols: the number of cols in resultant matrix
         */

        double[][] output = new double[nrows][ncols];
        for (int i = 0; i < nrows; i++)
            for (int j = 0; j < ncols; j++)
                output[i][j] = MatrixOps.max(
                    MatrixOps.slice(input, i * 2, i * 2 + 1, j * 2, j * 2 + 1)
                );
        return output;
    }

    public double[][][] forwardProp(double[][][] input) {
        /* forward propagation
         * : params
         * : input: the input image 3d matrix to pool over 
         */
        this.cacheInput = input;
        int nfilters = input.length,
            nrows = input[0].length / poolSize,
            ncols = input[0][0].length / poolSize;
        double[][][] output = new double[nfilters][nrows][ncols];

        for (int i = 0; i < nfilters; i++)
            output[i] = pool(input[i], nrows, ncols);

        this.cacheOutput = output;
        return output;
    }

    public double[][][] backwardProp(double[][][] derivLwrtOut) {
        /* backward propagation
         * params:
         * : input: the derivative of loss wrt outputs of this layer
         */
        double[][][] derivLwrtInputs = 
            new double[cacheInput.length][cacheInput[0].length][cacheInput[0][0].length];
        
        for (int i = 0; i < cacheOutput.length; i++)
            for (int j = 0; j < cacheOutput[0].length; j++)
                for (int k = 0; k < cacheOutput[0][0].length; k++) {
                    double[][] slice = MatrixOps.slice(
                        cacheInput[i], j * 2, j * 2 + 1, k * 2, k * 2 + 1
                    );
                    for (int r = 0; r < slice.length; r++)
                        for (int c = 0; c < slice[0].length; c++) {
                            double diff = Math.abs(slice[r][c] - cacheOutput[i][j][k]);
                            if (diff < doubleZero)
                                derivLwrtInputs[i][j * 2 + r][k * 2 + c] = derivLwrtOut[i][j][k];
                        }
                }
        return derivLwrtInputs;
    }
}
