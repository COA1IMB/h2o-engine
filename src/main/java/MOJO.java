import hex.genmodel.MojoModel;
import hex.genmodel.easy.EasyPredictModelWrapper;
import hex.genmodel.easy.RowData;
import hex.genmodel.easy.exception.PredictException;
import hex.genmodel.easy.prediction.BinomialModelPrediction;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MOJO
{

   public void setData(ArrayList<List<String>> data)
   {
      this.data = data;
   }

   private ArrayList<List<String>> data;

   public void test()
   {
      EasyPredictModelWrapper model = null;

      try
      {
         model = new EasyPredictModelWrapper(MojoModel.load("C:\\Development\\anaomaly\\src\\main\\resources\\gbm_28fd163e_8bcb_4510_a717_7f07c9ab0865.zip"));
      }
      catch (IOException e)
      {
         e.printStackTrace();
      }

      BinomialModelPrediction p = null;
      List<RowData> rowList = new ArrayList<RowData>();

      data.forEach((temp) -> {
         int i = 1;
         RowData row = new RowData();
         row.put("C1", temp.get(0));
         row.put("C2", temp.get(1));
         row.put("C3", temp.get(2));
         row.put("C4", temp.get(3));
         row.put("C5", temp.get(4));
         row.put("C6", temp.get(5));
         row.put("C7", temp.get(6));
         row.put("C8", temp.get(7));
         row.put("C9", temp.get(8));
         row.put("C10", temp.get(9));
         row.put("C11", temp.get(10));
         row.put("C12", temp.get(11));
         row.put("C13", temp.get(12));
         row.put("C14", temp.get(13));
         row.put("C15", temp.get(14));
         row.put("C16", temp.get(15));
         row.put("C17", temp.get(16));
         row.put("C18", temp.get(17));
         row.put("C19", temp.get(18));
         row.put("C20", temp.get(19));
         row.put("C21", temp.get(20));
         row.put("C22", temp.get(21));
         row.put("C23", temp.get(22));
         row.put("C24", temp.get(23));
         row.put("C25", temp.get(24));
         row.put("C26", temp.get(25));
         row.put("C27", temp.get(26));
         row.put("C28", temp.get(27));
         row.put("C29", temp.get(28));
         row.put("C30", temp.get(29));
         row.put("C31", temp.get(30));
         rowList.add(row);
      });

      for (RowData temp : rowList)
      {
         try
         {
            p = model.predictBinomial(temp);


            if(p.label.equals("1"))
            {
               System.out.println("Label (aka prediction) is: " + p.label);
               System.out.print("Class probabilities: ");
               for (int i = 0; i < p.classProbabilities.length; i++)
               {
                  if (i > 0)
                  {
                     System.out.print(",");
                  }
                  System.out.print(p.classProbabilities[i]);
               }
               System.out.println("");
            }
         }
         catch (PredictException e)
         {
            e.printStackTrace();
         }
      }
      System.out.println("Done");
   }
}
