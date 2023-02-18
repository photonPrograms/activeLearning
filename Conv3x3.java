public class Conv3x3 {
    /* a convolution layer with 3x3 filters with stride = 1
     * and valid padding for
     * both forward and backward passes
     */

    public double[][][] filters;
    public int numFilters;
    public double learningRate;
    public double[][] cacheInput;

    public Conv3x3(int numFilters) {
        this.numFilters = numFilters;
        this.learningRate = 5e-3;
        this.cacheInput = null;

        // initialize the filters randomly
        filters = new double[numFilters][3][3];
        for (int i = 0; i < numFilters; i++)
            filters[i] = MatrixOps.getRandom(3, 3);
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public double[][] convolve(double[][] input, int filterIndex, int nrows, int ncols) {
        /* 2d convolution operation
         * : params:
         * : input: the input image matrix
         * : filterIndex: the index of the filter being used
         * : nrows: the resultant number of rows
         * : ncols: the resultant number of columns 
         */
        double[][] outputSlice = new double[nrows][ncols];
        for (int i = 1; i < nrows; i++)
            for (int j = 1; j < ncols; j++)
                outputSlice[i][j] = MatrixOps.dot(
                    MatrixOps.slice(input, i - 1, i + 1, j - 1, j + 1),
                    filters[filterIndex]
                );
        return outputSlice;
    }
    
    public double[][][] forwardProp(double[][] input) {
        /* forward pass
         * : params
         * : input: the input image matrix
         */
        this.cacheInput = input;
        int nrows = input.length - 1, ncols = input[0].length - 1;
        double[][][] output = new double[numFilters][nrows][ncols];

        for (int i = 0; i < numFilters; i++)
            output[i] = convolve(input, i, nrows, ncols);

        return output;
    }

    public void backwardProp(double[][][] derivLwrtOut) {
        /* backward propagation
         * : params
         * : derivLwrtOut: the derivative of loss wrt layer outputs
         */

        double[][][] derivLwrtFilters = 
            new double[filters.length][filters[0].length][filters[0][0].length];
        int nrows = cacheInput.length - 1, ncols = cacheInput[0].length - 1;

        for (int i = 1; i < nrows; i++)
            for (int j = 1; j < ncols; j++)
                for (int f = 0; f < numFilters; f++)
                    derivLwrtFilters[f] = MatrixOps.add(
                        derivLwrtFilters[f],
                        MatrixOps.mult(
                            MatrixOps.slice(cacheInput, i - 1, i + 1, j - 1, j + 1),
                            derivLwrtOut[f][i - 1][j - 1]
                        )
                    );
        
        for (int f = 0; f < numFilters; f++)
            filters[f] = MatrixOps.add(
                filters[f],
                MatrixOps.mult(derivLwrtFilters[f], -this.learningRate)
            );
    }
}
