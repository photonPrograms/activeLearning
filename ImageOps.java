import java.util.*;
import java.io.*;
import java.awt.image.*;
import javax.imageio.*;

public class ImageOps {
    /* for various image operations on the png mnist images
     * such as reading and conversion to matrix
     */

     public static BufferedImage openImage(String filePath) throws IOException {
        /* open the image with the given file name
         * params:
         * : filePath: the path relative to source code directory
         */

         BufferedImage image = ImageIO.read(new File(filePath));
         return image;
     }


     public static double[][] convertImageToMatrix(BufferedImage image) {
        /* convert the image to a 2d array
         * params:
         * : image: the BufferedImage to be converted 
         */
        int rows = image.getWidth(), cols = image.getHeight();

        // a vector of pixels
        int[] imageVector = image.getRGB(0, 0, rows, cols, null, 0, rows);
        
        double[][] matrix = new double[rows][cols];
        int i = 0, j = 0;
        for (int pixel: imageVector) {
            // normalized to [-0.5, 0.5]
            matrix[i][j] = (double) (((int) pixel >> 16 & 0xff)) / 255.0 - 0.5;
            j++;
            if (j == cols) {
                j = 0;
                i++;
            }
        }

        return matrix;
     }

     public static double[][] getImageMatrix(String filePath) throws IOException {
        /* encapsulates openImage() and getImageMatrix()
         * params:
         * : filePath: the relative address of the image file
         */
        return convertImageToMatrix(openImage(filePath));
     }
     
     public static Image getImage(String filePath) throws IOException {
        /* open an image file and get the corresponding image object
         * params:
         * : filePath: the relative address of the image file
         */

        Image img = new Image();
        img.matrix = convertImageToMatrix(openImage(filePath));
        String[] words = filePath.split("/");
        int wordsLen = words.length;
        int label = Integer.parseInt(words[wordsLen - 2]);
        String uid = String.format(
            "%s_%d_%s", words[wordsLen - 3], label, words[wordsLen - 1].split("\\.")[0]
        );
        img.label = label;
        img.uid = uid;
        return img;
     }

     public static ArrayList<Image>
      getImages(String parentDir, int quantity, int numClasses) throws IOException {
        /* get a certain number of images from parent directory
         * and shuffle them
         * params:
         * parentDir: test or train directory
         * quantity: the number of images to be obtained (total)
         * numClasses: the total number of classes
         */

        int imagesPerClass = quantity / numClasses;
        ArrayList<Image> images = new ArrayList<>();
        
        for (int i = 0; i < numClasses; i++)
            for (int j = 1; j <= imagesPerClass; j++) {
                String currPath = String.format(
                    "%s/%d/%d.png", parentDir, i, j
                );
                Image currImage = getImage(currPath);
                images.add(currImage);
            }

        Collections.shuffle(images);
        return images;
     }

     public static ArrayList<Image> 
        getImages(String parentDir, int quantity, int totalStartIndex, int numClasses) 
        throws IOException {
        int imagesPerClass = quantity / numClasses;
        int startIndex = totalStartIndex / numClasses;
        ArrayList<Image> images = new ArrayList<>();

        for (int i = 0; i < numClasses; i++)
            for (int j = startIndex + 1; j <= imagesPerClass + startIndex; j++) {
                String currPath = String.format(
                    "%s/%d/%d.png", parentDir, i, j
                );
                Image currImage = getImage(currPath);
                images.add(currImage);
            }
        
        Collections.shuffle(images);
        return images;
     }

     public static ArrayList<ArrayList<Image>> splitList(ArrayList<Image> images, double splitRatio) {
        /* split a list of images into two 
         * params:
         * : images: the list of images to be split
         * : splitRatio: the first half as fraction of total 
         */

        int firstUpperLim = (int) Math.floor(splitRatio * images.size());
        ArrayList<ArrayList<Image>> partitions = new ArrayList<>();
        ArrayList<Image> currList = new ArrayList<>();
        for (int i = 0; i < firstUpperLim; i++)
            currList.add(images.get(i));
        partitions.add(currList);
        currList = new ArrayList<>();
        for (int i = firstUpperLim; i < images.size(); i++)
            currList.add(images.get(i));
        partitions.add(currList);
        return partitions;
     }

     public static ArrayList<Image> combineLists(ArrayList<Image> a, ArrayList<Image> b) {
        /* combine two lists and shuffle them
         * params:
         * a & b: the lists to be combined
         */
        ArrayList<Image> combined = new ArrayList<>();
        for (int i = 0; i < a.size(); i++)
            combined.add(a.get(i));
        for (int i = 0; i < b.size(); i++)
            combined.add(b.get(i));
        Collections.shuffle(combined);
        return combined;
     }
}