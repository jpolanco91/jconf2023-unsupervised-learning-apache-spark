package com.jconf2023.unsupervisedlearning;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import static org.apache.spark.sql.functions.col;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.OneHotEncoderModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.MinMaxScalerModel;
import org.apache.spark.ml.feature.PCA;
import org.apache.spark.ml.feature.PCAModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.types.*;
import static org.apache.spark.sql.types.DataTypes.*;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.ArrayList;

// Dataset utilizado: https://www.kaggle.com/datasets/sriramm2010/uci-bike-sharing-data
public class App 
{
    public static void main( String[] args )
    {
        // Importamos el dataset.
        String datasetPath = "/Users/juanjpolanco/Documents/jconf2023-unsupervised-learning/dataset/bike_sharing_dataset/day.csv";
        SparkSession spark = SparkSession.builder().master("local").appName("unsupervised_learning").getOrCreate();

        // Usamos el DataFrame API.
        Dataset<Row> df = spark.read().option("header", "true").option("inferSchema", "true").csv(datasetPath);
        df.show();

        /**
         * Explicacion de los features dataset
         * - instant: record index
         * - dteday : date
         * - season : season (1:spring, 2:summer, 3:fall, 4:winter).
         * - yr : year (0: 2011, 1:2012)
         * - mnth : month ( 1 to 12).
         * - hr : hour (0 to 23).
         * - holiday : weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule).
         * - weekday : day of the week.
         * - workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
         * - weathersit : 
		        - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
		        - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
		        - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
		        - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
           - temp : Normalized temperature in Celsius. The values are divided to 41 (max)
           - atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max).
           - hum: Normalized humidity. The values are divided to 100 (max).
           - windspeed: Normalized wind speed. The values are divided to 67 (max).
           - casual: count of casual users.
           - registered: count of registered users.
           - cnt: count of total rental bikes including both casual and registered
         */

        /** PREPROCESAMIENTO */

        // Removeremos la primera y segunda columna del dataset (instant y dteday) ya que no seran necesarias para el modelo.
        Dataset<Row> trimmedDataset = df.drop(col("instant"), col("dteday"));
        trimmedDataset.show();


        // Exploracion de la data.
        trimmedDataset.describe().show();

        // Aplicaremos OneHot encoding a la data categorica para llevarla a numerica.
        // Dicha data categorica son los features: season, yr, mnth, holiday, weekday, workingday y weathersit.
        OneHotEncoder encoder = new OneHotEncoder()
            .setInputCols(new String[] {"season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"})
            .setOutputCols(new String[] {"enc_season", "enc_yr", "enc_mnth", "enc_holiday", "enc_weekday", "enc_workingday", "enc_weathersit"})
            .setDropLast(false);
        
        OneHotEncoderModel encModel = encoder.fit(trimmedDataset);
        Dataset<Row> encodedDataset = encModel.transform(trimmedDataset);

        Dataset<Row> nonCategoricalData = trimmedDataset.select(col("temp"), col("atemp"), col("hum"), col("windspeed"), col("casual"), col("registered"), col("cnt"));

        Dataset<Row> encodedSeasonColumn = encodedDataset.select(col("enc_season"));
        Dataset<Row> encodedYrColumn = encodedDataset.select(col("enc_yr"));
        Dataset<Row> encodedMnthColumn = encodedDataset.select(col("enc_mnth"));
        Dataset<Row> encodedHolidayColumn = encodedDataset.select(col("enc_holiday"));
        Dataset<Row> encodedWeekdayColumn = encodedDataset.select(col("enc_weekday"));
        Dataset<Row> encodedWorkingDayColumn = encodedDataset.select(col("enc_workingday"));
        Dataset<Row> encodedWeatherSitColumn = encodedDataset.select(col("enc_weathersit"));

        Dataset<Row> transformedSeasonColumn = oneHotToDfColumn(encodedSeasonColumn, "enc_season", new String[]{"1", "2", "3", "4"}, spark);
        Dataset<Row> transformedYrColumn = oneHotToDfColumn(encodedYrColumn, "enc_yr", new String[]{"0", "1"}, spark);
        Dataset<Row> transformedMnthColumn = oneHotToDfColumn(encodedMnthColumn, "enc_mnth", new String[]{"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"}, spark);
        Dataset<Row> transformedHolidayColumn = oneHotToDfColumn(encodedHolidayColumn, "enc_holiday", new String[]{"0", "1"}, spark);
        Dataset<Row> transformedWeekdayColumn = oneHotToDfColumn(encodedWeekdayColumn, "enc_weekday", new String[]{"0", "1", "2", "3", "4", "5", "6"}, spark);
        Dataset<Row> transformedWorkingDayColumn = oneHotToDfColumn(encodedWorkingDayColumn, "enc_workingday", new String[]{"0", "1"}, spark);
        Dataset<Row> transformedWeatherSitColumn = oneHotToDfColumn(encodedWeatherSitColumn, "enc_weathersit", new String[]{"1", "2", "3", "4"}, spark);


        // Uniendo los datasets nuevamente y convirtiendolos a vector a traves de VectorAssembler.
        Dataset<Row> dataf = transformedSeasonColumn.crossJoin(transformedYrColumn);
        Dataset<Row> dataf2 = dataf.crossJoin(transformedMnthColumn);
        Dataset<Row> dataf3 = dataf2.crossJoin(transformedHolidayColumn);
        Dataset<Row> dataf4 = dataf3.crossJoin(transformedWeekdayColumn);
        Dataset<Row> dataf5 = dataf4.crossJoin(transformedWorkingDayColumn);
        Dataset<Row> dataf6 = dataf5.crossJoin(transformedWeatherSitColumn);
        Dataset<Row> finalEncodedDataset = dataf6.crossJoin(nonCategoricalData);

        finalEncodedDataset.show(270);

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{
                    "temp", 
                    "atemp",
                    "hum",
                    "windspeed",
                    "casual",
                    "registered",
                    "cnt",
                    "enc_season1",
                    "enc_season1",
                    "enc_season2",
                    "enc_season3",
                    "enc_season4",
                    "enc_yr0",
                    "enc_yr1",
                    "enc_mnth1",
                    "enc_mnth2",
                    "enc_mnth3",
                    "enc_mnth4",
                    "enc_mnth5",
                    "enc_mnth6",
                    "enc_mnth7",
                    "enc_mnth8",
                    "enc_mnth9",
                    "enc_mnth10",
                    "enc_mnth11",
                    "enc_mnth12",
                    "enc_holiday0",
                    "enc_holiday1",
                    "enc_weekday0",
                    "enc_weekday1",
                    "enc_weekday2",
                    "enc_weekday3",
                    "enc_weekday4",
                    "enc_weekday5",
                    "enc_weekday6",
                    "enc_workingday0",
                    "enc_workingday1",
                    "enc_weathersit1",
                    "enc_weathersit2",
                    "enc_weathersit3",
                    "enc_weathersit4"
                })
                .setOutputCol("features");
        
        Dataset<Row> vectorizedFeatures = assembler.transform(finalEncodedDataset);
        vectorizedFeatures.select(col("features")).show(false);

        // Usamos MinMax scaler para escalar la data continua entre 0 y 1 ya que el modelo de K-Means asi lo requiere.
        MinMaxScaler scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures");
        MinMaxScalerModel scalerModel = scaler.fit(vectorizedFeatures);
        Dataset<Row> scaledFeatures = scalerModel.transform(vectorizedFeatures);
        Dataset<Row> scaledData = scaledFeatures.select(col("scaledFeatures"));
        Dataset<Row> scaledDataTrimmed = (Dataset<Row>)scaledData.head(200);
        scaledDataTrimmed.show();

        /** Entrenando modelo de K-Means clustering. */

        // Un algoritmo de Clustering llamado K-Means Clustering con 3 clusters.

        KMeans kmeans = new KMeans().setK(2).setSeed(1L);
        KMeansModel model = kmeans.fit(scaledDataTrimmed);

        // Hacemos las predicciones de los clusters a los que pertenecen los data point.
        Dataset<Row> predictions = model.transform(scaledDataTrimmed);

        ClusteringEvaluator evaluator = new ClusteringEvaluator();

        // Calculamos el silhouette score.
        double silhouetteScore = evaluator.evaluate(predictions);
        System.out.println("Silhouette score = " + silhouetteScore);

        // Calculamos los centros de cada cluster.
        Vector[] centers = model.clusterCenters();
        System.out.println("Cluster Centers: ");
        for (Vector center: centers) {
            System.out.println(center);
        }

        // Detenemos la aplicacion y asi finalizamos la sesion que se almacena en memoria.
        spark.stop();
    }

    // OneHot vector to Vector function.
    public static Dataset<Row> oneHotToDfColumn(Dataset<Row> columnDf, String columnName, String[] values, SparkSession sp) {
        StructField[] schemaFields = new StructField[values.length];

        for (int i = 0; i < values.length; i++) {
            schemaFields[i] = new StructField(columnName + values[i], DataTypes.DoubleType, false, Metadata.empty());
        }

        // Creando un nuevo schema para las nuevas columnas codificadas con OneHot.
        StructType schema = new StructType(schemaFields);

        // Reestructurando los nuevos valores codificados.
        Iterator<Row> it = columnDf.toLocalIterator();
        List<Row> data = new ArrayList<>();

        int initialValue = Integer.parseInt(values[0]);

        while(it.hasNext()) {
            Double[] rowValue = new Double[values.length];
            Row currentRow = it.next();

            for (int i = initialValue; i < (values.length + initialValue); i++) {
                Vector vt = (Vector)currentRow.get(0);
                Double vtValue;
                if (i == values.length) {
                    vtValue = vt.apply(i-initialValue);
                } else {
                    vtValue = vt.apply(i);
                }
                rowValue[i-initialValue] = vtValue;
            }

            data.add(RowFactory.create(rowValue));
        }
        
        Dataset<Row> convertedData = sp.createDataFrame(data, schema);

        return convertedData;
    }
}
