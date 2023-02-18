import java.util.*;

public class AggregateResults {
    /* to aggregate (mean/std dev) the trial results */
    public String frac; // the fraction of samples
    public ArrayList<Double> trainResults; // accuracy on training data
    public ArrayList<Double> totalTrainResults; // accuracy on total training data
    public ArrayList<Double> testResults; // accuracy on testing data

    // mean and standard deviation for the various portions
    public double meanTrain, devTrain, meanTotal, devTotal, meanTest, devTest;

    public AggregateResults(String frac) {
        this.frac = frac;
        trainResults = new ArrayList<>();
        totalTrainResults = new ArrayList<>();
        testResults = new ArrayList<>();
    }

    public static ArrayList<String> aggregateResults(ArrayList<String> results) {
        /* aggregate the given results - frac, tr, tot, test quadruplets */
        HashMap<String, AggregateResults> resultMap = new HashMap<>();
        for (String result: results) {
            String[] resultArr = result.split(", ");
            if (!resultMap.containsKey(resultArr[0]))
                resultMap.put(resultArr[0], new AggregateResults(resultArr[0]));
            AggregateResults currEntry = resultMap.get(resultArr[0]);
            currEntry.trainResults.add(Double.parseDouble(resultArr[1]));
            currEntry.totalTrainResults.add(Double.parseDouble(resultArr[2]));
            currEntry.testResults.add(Double.parseDouble(resultArr[3]));
        }
        ArrayList<String> stats = new ArrayList<>();
        for (Map.Entry<String, AggregateResults> me: resultMap.entrySet()) {
            me.getValue().meanTrain = getAverage(me.getValue().trainResults);
            me.getValue().devTrain = getStdDev(me.getValue().trainResults);
            me.getValue().meanTotal = getAverage(me.getValue().totalTrainResults);
            me.getValue().devTotal = getStdDev(me.getValue().totalTrainResults);
            me.getValue().meanTest = getAverage(me.getValue().testResults);
            me.getValue().devTest = getStdDev(me.getValue().testResults);

            stats.add(String.format(
                "%s %.2f %.2f %.2f %.2f %.2f %.2f",
                me.getKey(),
                me.getValue().meanTrain, me.getValue().meanTotal, me.getValue().meanTest,
                me.getValue().devTrain, me.getValue().devTotal, me.getValue().devTest
            ));
        }
        Collections.sort(stats);
        return stats;
    }

    public static double getAverage(ArrayList<Double> arrList) {
        /* get the average of the list */
        if (arrList.size() <= 0) {
            System.out.println("Attempt to calculate average with empty list!");
            return 0;
        }
        double sum = 0;
        for (Double num: arrList)
            sum += num;
        return sum / arrList.size();
    }

    public static double getStdDev(ArrayList<Double> arrList) {
        /* get sample standard deviation the given list */
        if (arrList.size() <= 1) {
            System.out.println("Attempt to calculate std dev with <= 1 entries.");
            return 0;
        }
        double sum = 0, avg = getAverage(arrList);
        for (Double num: arrList)
            sum += Math.pow(num - avg, 2);
        return Math.sqrt(Math.abs(sum / (arrList.size() - 1)));
    }
}
