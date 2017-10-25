import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.ListStringRecordReader;
import org.datavec.api.split.ListStringSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.deeplearning4j.nn.conf.Updater.RMSPROP;

public class App
{

   public static void main(String[] args)
   {
      //MOJO mo = new MOJO();
      //mo.setData(getDataAsList("C:\\Development\\anaomaly\\src\\main\\resources\\creditcard.csv"));
      //mo.test();

      String JDBC_DRIVER = "com.mysql.jdbc.Driver";
      String DB_URL = "jdbc:mysql://localhost:8040/test?autoReconnect=true&useSSL=false";

      //  Database credentials
      String USER = "root";
      String PASS = "admin";

      Connection conn = null;
      Statement stmt = null;
      try
      {
         //STEP 2: Register JDBC driver
         Class.forName("com.mysql.jdbc.Driver");

         //STEP 3: Open a connection
         System.out.println("Connecting to database...");
         conn = DriverManager.getConnection(DB_URL, USER, PASS);

         //STEP 4: Execute a query
         System.out.println("Creating statement...");
         stmt = conn.createStatement();
         String sql;
         sql = "SELECT * from cats";
         ResultSet rs = stmt.executeQuery(sql);

         //STEP 5: Extract data from result set
         while (rs.next())
         {
            //Retrieve by column name
            int id = rs.getInt("id");
            String age = rs.getString("name");
            String first = rs.getString("owner");
            String last = rs.getString("birth");

            //Display values
            System.out.print("ID: " + id);
            System.out.print(", Name: " + age);
            System.out.print(", Owner: " + first);
            System.out.println(", Birth: " + last);
         }
         //STEP 6: Clean-up environment
         rs.close();
         stmt.close();
         conn.close();
      }
      catch (SQLException se)
      {
         //Handle errors for JDBC
         se.printStackTrace();
      }
      catch (Exception e)
      {
         //Handle errors for Class.forName
         e.printStackTrace();
      }
      finally
      {
         //finally block used to close resources
         try
         {
            if (stmt != null)
               stmt.close();
         }
         catch (SQLException se2)
         {
         }// nothing we can do
         try
         {
            if (conn != null)
               conn.close();
         }
         catch (SQLException se)
         {
            se.printStackTrace();
         }//end finally try
      }//end try
      System.out.println("Goodbye!");
   }

   public static ArrayList<List<String>> getDataAsList(String file)
   {

      ArrayList<List<String>> data = new ArrayList<>();
      String sCurrentLine = "";
      String learnFilePath = file;

      try (BufferedReader br = new BufferedReader(new FileReader(learnFilePath)))
      {

         while ((sCurrentLine = br.readLine()) != null)
         {
            String[] parts1 = sCurrentLine.split(",");
            List<String> data2 = Arrays.asList(parts1);
            data.add(data2);
         }
      }
      catch (IOException e)
      {

      }
      return data;
   }

   private static void train()
   {

      double bestModelScore = 999999999;

      MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
         .seed(12345)
         .iterations(1)
         .weightInit(WeightInit.XAVIER)
         .activation(Activation.RELU)
         .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
         .regularization(true).l2(0.0001)
         .list()
         .layer(0, new DenseLayer.Builder().nIn(29).nOut(250).updater(RMSPROP)
            .build())
         .layer(1, new DenseLayer.Builder().nIn(250).nOut(15).updater(RMSPROP)
            .build())
         .layer(2, new DenseLayer.Builder().nIn(15).nOut(250).updater(RMSPROP)
            .build())
         .layer(3, new OutputLayer.Builder().nIn(250).nOut(29).updater(RMSPROP)
            .lossFunction(LossFunctions.LossFunction.MSE)
            .build())
         .pretrain(true).backprop(true)
         .build();

      MultiLayerNetwork net = new MultiLayerNetwork(conf);
      net.init();
      ScoreIterationListener x = new ScoreIterationListener(10);
      net.setListeners(x);
      UIServer uiServer = UIServer.getInstance();
      StatsStorage statsStorage = new InMemoryStatsStorage();
      net.setListeners(new StatsListener(statsStorage));
      uiServer.attach(statsStorage);

      List<List<String>> dataOriginal = getDataAsList("src\\main\\resources\\creditcard.csv");
      List<List<String>> dataModified = new ArrayList<>();

      dataOriginal.forEach((temp) -> {
         List<String> dataTemp = new ArrayList<>();
         if (temp.get(30).equals("0"))
         {
            for (int i = 1; i < temp.size(); i++)
            {
               dataTemp.add(temp.get(i));
            }
            dataModified.add(dataTemp);
         }
      });

      List<INDArray> featuresTrain = new ArrayList<>();

      DataSetIterator iter = null;

      try (RecordReader rr = new ListStringRecordReader())
      {
         rr.initialize(new ListStringSplit(dataModified));
         iter = new RecordReaderDataSetIterator(rr, 1000, 29, 2);
      }
      catch (Exception e)
      {

      }

      while (iter.hasNext())
      {
         DataSet ds = iter.next();
         featuresTrain.add(ds.getFeatureMatrix());
      }
      //Train model:
      int nEpochs = 300;
      for (int epoch = 0; epoch < nEpochs; epoch++)
      {
         for (INDArray data : featuresTrain)
         {
            net.fit(data, data);
         }

         double epocheScore = net.score();
         if (epocheScore < bestModelScore)
         {
            bestModelScore = epocheScore;
            try
            {
               ModelSerializer.writeModel(net, "src\\NeuralNetwork.zip", true);
            }
            catch (Exception e)
            {

            }
            System.out.println("New best model at epoche: " + epoch + " - Score is " + epocheScore);
         }
         else
         {
            System.out.println("No better model in epcohe: " + epoch + " - Score is " + epocheScore);
         }
      }

   }

   private static double getEpocheScore(MultiLayerNetwork net, DataSetIterator iter)
   {

      double score = 0;
      List<INDArray> featuresTest = new ArrayList<>();
      iter.reset();


      while (iter.hasNext())
      {
         DataSet ds = iter.next();
         featuresTest.add(ds.getFeatureMatrix());
      }
      for (int i = 0; i < featuresTest.size(); i++)
      {
         INDArray testData = featuresTest.get(i);
         int nRows = testData.rows();

         for (int j = 0; j < nRows; j++)
         {
            INDArray example = testData.getRow(j);
            score = score + net.score(new DataSet(example, example));
         }
      }
      return score;
   }
}
