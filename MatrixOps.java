public class MatrixOps {
    /* a numpy like homebrewed library
     * for simple operations involving 2d matrices 
     */

    public static double[][] getRandom(int nrows, int ncols) {
        /* get a 2d matrix with random numbers b/w 0 and 1
         * params:
         * : nrows: the number of rows
         * : ncols: the number of columns
         */
        double[][] matrix = new double[nrows][ncols];
        for (int i = 0; i < nrows; i++)
            for (int j = 0; j < ncols; j++)
                matrix[i][j] = Math.random();
        return matrix;
    }

    public static double[][] zeros(int nrows, int ncols) {
        /* get a 2d zero matrix
         * params:
         * : nrows: the number of rows
         * : ncols: the number of columns
         */

        double[][] matrix = new double[nrows][ncols];
        for (int i = 0; i < nrows; i++)
            for (int j = 0; j < ncols; j++)
                matrix[i][j] = 0;
        return matrix;
    }

    public static double[][] slice(double[][] M, 
                                    int beginRow, int endRow,
                                    int beginCol, int endCol) {
        /* get a slice of the matrix M
         * params:
         * : M: the matrix to obtain slice from
         * : beginRow: the row index at which to begin
         * : endRow: the row index at which to end
         * : beginCol: the column index at which to begin
         * : endCol: the column index at which to end
         */

        double[][] sliceOfM = new double[endRow - beginRow + 1][endCol - beginCol + 1];
        for (int i = 0; i < sliceOfM.length; i++)
            for (int j = 0; j < sliceOfM[0].length; j++)
                sliceOfM[i][j] = M[beginRow + i][beginCol + j];
        return sliceOfM;
    }

    public static double dot(double[][] A, double[][] B) {
        /* get dot product of two m x n matrices
         * : params
         * : A: first operand matrix
         * : B: second operand matrix
         */
        
        double dotProduct = 0;
        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < A[0].length; j++)
                dotProduct += A[i][j] * B[i][j];
        return dotProduct;
    }
    
    public static double max(double[][] M) {
        /* get maximum element over the matrix
         * : params
         * : M: the matrix 
         */
        double maxElement = Double.MIN_VALUE;
        for (int i = 0; i < M.length; i++)
            for (int j = 0; j < M[0].length; j++)
                maxElement = Math.max(maxElement, M[i][j]);
        return maxElement;
    }

    public static int[] argmax(double[][] M) {
        /* get the index of maximum element of the matrix
         * : params
         * : M: the matrix
         */
        double maxElement = Double.MIN_VALUE;
        int[] indices = new int[2];
        for (int i = 0; i < M.length; i++)
            for (int j = 0; j < M[0].length; j++)
                if (M[i][j] > maxElement) {
                    maxElement = M[i][j];
                    indices = new int[] {i, j};
                }
        return indices;
    }

    public static double sum(double[][] M) {
        /* get the sum of all elements in matrix
         * params:
         * : M: the matrix
         */
        double sumOfElements = 0;
        for (int i = 0; i < M.length; i++)
            for (int j = 0; j < M[0].length; j++)
                sumOfElements += M[i][j];
        return sumOfElements;
    }

    public static double dist(double[][] A, double[][] B) {
        /* get the euclidean distance between A and B
         * params:
         * A & B: the 2d matrices (of equal dimensions)
         */

        int nrows = A.length, ncols = A[0].length;
        double dist = 0;
        for (int i = 0; i < nrows; i++)
            for (int j = 0; j < ncols; j++)
                dist += Math.pow(A[i][j] - B[i][j], 2);
        return Math.sqrt(dist);
    }

    public static double[][] mult(double[][] M, double k) {
        /* return the result of scalar multiplication k*M
         * params:
         * : M: the matrix
         * : k: the scalar
         */

        int nrows = M.length, ncols = M[0].length;
        double[][] output = new double[nrows][ncols];
        for (int i = 0; i < nrows; i++)
            for (int j = 0; j < ncols; j++)
                output[i][j] = M[i][j] * k;
        return output;
    }

    public static double[][] add(double[][] A, double[][] B) {
        /* elementwise matrix addition
         * params:
         * : A: the first 2d matrix operand
         * : B: the second 2d matrix operand
         */

        int nrows = A.length, ncols = A[0].length;
        double[][] output = new double[nrows][ncols];
        for (int i = 0; i < nrows; i++)
            for (int j = 0; j < ncols; j++)
                output[i][j] = A[i][j] + B[i][j];
        return output;
    }

    public static double[][] prod(double[][] A, double[][] B) {
        /* matrix multiplication of 2d matrices
         * params:
         * : A: the first operand
         * : B: the second operand
         */
        int nrows = A.length, ncols = B[0].length, ncommon = A[0].length;
        double[][] output = new double[nrows][ncols];
        for (int i = 0; i < nrows; i++)
            for (int j = 0; j < ncols; j++)
                for (int k = 0; k < ncommon; k++)
                    output[i][j] += A[i][k] * B[k][j];
        return output;
    }

    public static double[][] transpose(double[][] M) {
        /* take transpose of a 2d matrix
         * params:
         * : M: the 2d matrix operand
         */
        int nrows = M[0].length, ncols = M.length;
        double[][] output = new double[nrows][ncols];
        for (int i = 0; i < nrows; i++)
            for (int j = 0; j < ncols; j++)
                output[i][j] = M[j][i];
        return output;
    }

    public static double[][] exp(double[][] M) {
        /* elementwise exponentiation with base e
         * params:
         * : M: the 2d matrix
         */

        int nrows = M.length, ncols = M[0].length;
        double[][] output = new double[nrows][ncols];
        for (int i = 0; i < nrows; i++)
            for (int j = 0; j < ncols; j++)
                output[i][j] = Math.exp(M[i][j]);
        return M;
    }

    public static double[][] flatten(double[][] M) {
        /* return a  1 x size vector after flattening a matrix
         * params:
         * : M: a 2d matrix
         */
        int nrows = M.length, ncols = M[0].length;
        double[][] vector = new double[1][nrows * ncols];
        
        int vectorIndex = 0;
        for (int i = 0; i < nrows; i++)
            for (int j = 0; j < ncols; j++)
                vector[0][vectorIndex++] = M[i][j];
        return vector; 
    }

    public static double[][] flatten(double[][][] M) {
        /* return a  1 x size vector after flattening a matrix
         * params:
         * : M: a 3d matrix
         */

        int nfilters = M.length, nrows = M[0].length, ncols = M[0][0].length;
        double[][] vector = new double[1][nfilters * nrows * ncols];
        
        int vectorIndex = 0;
        for (int f = 0; f < nfilters; f++)
            for (int i = 0; i < nrows; i++)
                for (int j = 0; j < ncols; j++)
                    vector[0][vectorIndex++] = M[f][i][j];
        return vector;
    }

    public static double[][][] get3D(double[][] vector, int[] shape) {
        /* get a 3d array from a vector (stored as a 2d matrix)
         * params:
         * : vector: the vector to be reshaped
         * : shape: the new 3d shape (length 3 array)
         */
        if (shape.length != 3) {
            System.out.println("Reshaping failed!!\n");
            return null;
        }
        double[][][] output = new double[shape[0]][shape[1]][shape[2]];
        int vectorIndex = 0;
        for (int f = 0; f < shape[0]; f++)
            for(int i = 0; i < shape[1]; i++)
                for (int j = 0; j < shape[2]; j++)
                    output[f][i][j] = vector[0][vectorIndex++];
        return output;
    }
}
