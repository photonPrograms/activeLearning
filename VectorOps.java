public class VectorOps {
    /* vector operations
     */

    public static int argmax(double[] vector) {
        /* the index for the maximum value in the vector
         * : params
         * : vector: the vector
         */

        int maxIndex = 0;
        double maxValue = Double.MIN_VALUE;
        for (int i = 0; i < vector.length; i++) {
            if (vector[i] > maxValue) {
                maxValue = vector[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
