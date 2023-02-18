public class Dense {
    /* a softmax activated fully connected layer
     * both forward and backward passes
     */

    public double[][] W;
    public double[][] b;
    public double learningRate;
    public int inputSize;
    public int outputSize;

    public double[][] cacheInput, cacheZ;
    public int[] cacheInputShape;

    public Dense(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        learningRate = 5e-3;
        cacheInput = null;
        cacheZ = null;
        cacheInputShape = null;

        W = MatrixOps.mult(
            MatrixOps.getRandom(inputSize, outputSize),
            (double) 1 / inputSize
        );
        b = MatrixOps.zeros(1, outputSize);
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public double[][] forwardProp(double[][][] input) {
        /* forward propagation
         * params:
         * : input: the input tensor from maxpooling layer
         */
        cacheInputShape = new int[] {
            input.length, input[0].length, input[0][0].length
        };
        double[][] inputVector = MatrixOps.flatten(input);
        cacheInput = inputVector;

        double[][] Z = MatrixOps.add(
            MatrixOps.prod(inputVector, W), b
        );
        cacheZ = Z;
        
        double[][] T = MatrixOps.exp(Z);
        double denominator = MatrixOps.sum(T);

        double[][] A = MatrixOps.mult(T, 1 / denominator);
        return A;
    }

    public double[][][] backwardProp(double[][] derivLwrtOut) {
        /* backward propagation
         * params:
         * : derivLwrtOut: the derivative of loss wrt outputs
         */

        double[][] derivLwrtT = new double[1][derivLwrtOut[0].length];

        double[][] T = MatrixOps.exp(cacheZ);
        double S = MatrixOps.sum(T);
        double[][] derivLwrtInputs = null;

        for (int i = 0; i < derivLwrtOut[0].length; i++) {
            double curr = derivLwrtOut[0][i];
            if (curr == 0)
                continue;
            
            double[][] derivOutwrtT = MatrixOps.mult(T, -T[0][i] / Math.pow(S, 2));
            derivOutwrtT[0][i] = T[0][i] / Math.pow(S, 2) * (S - T[0][i]);
            derivLwrtT = MatrixOps.mult(derivOutwrtT, curr);

            double[][] derivTwrtW = MatrixOps.transpose(cacheInput);
            double[][] derivTwrtInputs = W;
            double[][] derivLwrtW = MatrixOps.prod(derivTwrtW, derivLwrtT);
            derivLwrtInputs = MatrixOps.prod(
                derivTwrtInputs, MatrixOps.transpose(derivLwrtT)
            );
            double[][] derivLwrtB = derivLwrtT;

            W = MatrixOps.add(
                W, MatrixOps.mult(derivLwrtW, -this.learningRate)
            );
            b = MatrixOps.add(
                b, MatrixOps.mult(derivLwrtB, -this.learningRate)
            );
        }

        return MatrixOps.get3D(
            MatrixOps.transpose(derivLwrtInputs),
            cacheInputShape
        );
    }
}