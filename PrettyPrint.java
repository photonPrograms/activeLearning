import java.io.*;
import java.util.*;

public class PrettyPrint {
    /* formatting display of the results
     */

    public static final String epochAccFile = "./series/epochAcc.dat";
    public static final String epochLossFile = "./series/epochLoss.dat";
    public static final String stepAccFile = "./series/stepAcc.dat";
    public static final String stepLossFile = "./series/stepLoss.dat";

    public static final String trialFile = "./results/trial_";
    public static final String aggFile = "./results/agg_";

    public static void displayConfusionMatrix(int[][] confusionMatrix) {
        int nrows = confusionMatrix.length, ncols = confusionMatrix[0].length;
        String actualLabel = "Actual Classes" + "  ", predictedLabel = "Predicted Classes";

        System.out.println("CONFUSION MATRIX");
        for (int i = 0; i < ncols / 2 * 7 + 5; i++)
            System.out.print(" ");
        System.out.println(predictedLabel + "\n");

        for (int i = 0; i < actualLabel.length() + 6; i++)
            System.out.print(" ");
        for (int i = 0; i < ncols; i++)
            System.out.print(
                String.format(
                    "%5d  ", i
                )
            );
        System.out.println();

        for (int i = 0; i < actualLabel.length() + 6; i++)
            System.out.print(" ");
        for (int i = 0; i < ncols; i++)
            System.out.print("|------");
        System.out.println("|");

        

        for (int i = 0; i < nrows; i++) {
            if (i == 4)
                System.out.print(actualLabel);
            else
                for (int j = 0; j < actualLabel.length(); j++)
                    System.out.print(" ");
            System.out.print(
                String.format(
                    "%5d ", i
                )
            );
            for (int j = 0; j < ncols; j++)
                System.out.print(
                    String.format(
                        "|%5d ", confusionMatrix[i][j]
                    )
                );
            System.out.println("|");
            for (int j = 0; j < actualLabel.length() + 6; j++)
                System.out.print(" ");
            for (int j = 0; j < ncols; j++)
                System.out.print("|------");
            System.out.println("|");
        }
    }

    public static void saveSeries(
        ArrayList<Double> epochAcc, ArrayList<Double> epochLoss,
        ArrayList<Double> stepAcc, ArrayList<Double> stepLoss
    ) {
        /* saving to file the series of
         * params:
         * : epochAcc: model accuracy vs training epoch
         * : epochLoss: model loss vs training epoch
         * : stepAcc: model accuracy vs training step
         * : stepLoss: model loss vs training step
         */
        writeOneSeries(epochAcc, epochAccFile);
        writeOneSeries(epochLoss, epochLossFile);
        writeOneSeries(stepAcc, stepAccFile);
        writeOneSeries(stepLoss, stepLossFile);
    }

    public static void writeOneSeries(ArrayList<Double> series, String fileName) {
        /* write a single arraylist to a given file
         * params:
         * : series: the arraylist to be written
         * : fileName: the file path
         */
        PrintWriter outStream = null;
        try {
            outStream = new PrintWriter(new FileOutputStream(fileName));
            for (int i = 0; i < series.size(); i++) {
                outStream.println(i + " " + series.get(i));
            }
        }
        catch (IOException e) {
            System.err.println(e.getMessage());
        }
        catch (Exception e) {
            System.err.println(e.getMessage());
        }
        finally {
            outStream.close();
        }
    }

    public static void writeRawTrials(ArrayList<String> trialResults, String technique) {
        String fileName = trialFile + technique + ".csv";
        PrintWriter outStream = null;
        try {
            outStream = new PrintWriter(new FileOutputStream(fileName));
            for (int i = 0; i < trialResults.size(); i++) {
                outStream.println(trialResults.get(i));
            }
        }
        catch (IOException e) {
            System.err.println(e.getMessage());
        }
        catch (Exception e) {
            System.err.println(e.getMessage());
        }
        finally {
            outStream.close();
        }
    }

    public static void writeAggregateTrials(ArrayList<String> trialResults, String technique) {
        String fileName = aggFile + technique + ".csv";
        PrintWriter outStream = null;
        try {
            outStream = new PrintWriter(new FileOutputStream(fileName));
            for (int i = 0; i < trialResults.size(); i++) {
                outStream.println(trialResults.get(i));
            }
        }
        catch (IOException e) {
            System.err.println(e.getMessage());
        }
        catch (Exception e) {
            System.err.println(e.getMessage());
        }
        finally {
            outStream.close();
        }
    }

    public static ArrayList<String> openRawTrials(String technique) {
        String fileName = trialFile + technique + ".csv";
        Scanner inStream = null;
        ArrayList<String> results = new ArrayList<>();
        try {
            inStream = new Scanner(new FileInputStream(fileName));
            while (inStream.hasNextLine())
                results.add(inStream.nextLine());
        }
        catch (Exception e) {
            System.err.println(e.getMessage());
        }
        finally {
            inStream.close();
        }
        return results;
    }
}