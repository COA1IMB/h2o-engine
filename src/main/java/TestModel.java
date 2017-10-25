import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.ListStringRecordReader;
import org.datavec.api.split.ListStringSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

public class TestModel {
    public static void evaluate(){
        MultiLayerNetwork net = null;
        try {
            net = ModelSerializer.restoreMultiLayerNetwork("src\\NeuralNetwork.zip");
        } catch (Exception e) {

        }
        List<List<String>> dataOriginal = App.getDataAsList("src\\main\\resources\\creditcardTest.csv");
        List<List<String>> dataModified = new ArrayList<>();

        dataOriginal.forEach((temp) -> {
            List<String> dataTemp = new ArrayList<>();
            for (int i = 1; i < temp.size(); i++) {
                dataTemp.add(temp.get(i));
            }
            dataModified.add(dataTemp);
        });

        List<INDArray> featuresTest = new ArrayList<>();
        List<INDArray> labelsTest = new ArrayList<>();
        DataSetIterator iter = null;

        try (RecordReader rr = new ListStringRecordReader()) {
            rr.initialize(new ListStringSplit(dataModified));
            iter = new RecordReaderDataSetIterator(rr, 100, 29, 2);
        } catch (Exception e) {

        }

        int truePositive = 0;
        int trueNegatives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;
        int totalFraud = 0;
        int usualTransactions = 0;
        int totalTransactions = 0;
        double cutOffValue = 24.53385;
        double goodError = 0.0;
        double badError = 0.0;

        while (iter.hasNext()) {
            DataSet ds = iter.next();
            featuresTest.add(ds.getFeatureMatrix());
            INDArray indexes = Nd4j.argMax(ds.getLabels(),1); //Convert from one-hot representation -> index
            labelsTest.add(indexes);
        }
        for(int i = 0; i<featuresTest.size(); i++){
            INDArray testData = featuresTest.get(i);
            INDArray labels = labelsTest.get(i);
            int nRows = testData.rows();

            for( int j=0; j<nRows; j++){
                INDArray example = testData.getRow(j);
                int digit = (int)labels.getDouble(j);
                double score = net.score(new DataSet(example,example),false);

                // Add (score, example) pair to the appropriate list
                if (score > cutOffValue && digit == 1) {
                    truePositive++;
                    totalFraud++;
                    badError = badError + score;
                }
                if (score < cutOffValue && digit == 1) {
                    falsePositives++;
                    totalFraud++;
                    badError = badError + score;
                }
                if (score > cutOffValue && digit == 0) {
                    falseNegatives++;
                    usualTransactions++;
                    goodError = goodError + score;
                }
                if (score < cutOffValue && digit == 0) {
                    trueNegatives++;
                    usualTransactions++;
                    goodError = goodError + score;
                }
                totalTransactions++;
            }

        }
        System.out.println("Total Transactions: " + totalTransactions);
        System.out.println("Total Fraud: " + totalFraud);
        System.out.println("Total usual transactions: " + usualTransactions);
        System.out.println("Richtig erkannte Frauds: " + truePositive);
        System.out.println("Richtig erkannte nicht Frauds: " + trueNegatives);
        System.out.println("Nicht erkannte Frauds: " + falsePositives);
        System.out.println("Fälschlicherweiße als Fraud eingestuft : " + falseNegatives);
        System.out.println("GOOD MEAN: " + goodError/usualTransactions);
        System.out.println("BAD MEAN: " + badError/totalFraud);
    }
}
