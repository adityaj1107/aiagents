package dqn;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

// Helper class to build the DL4J network
public class QNetworkBuilder {

    public static MultiLayerNetwork buildQNetwork(int stateSize, int actionSize, int hiddenDim, double learningRate, long seed) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER) // Common weight initialization
                .updater(new Adam(learningRate)) // Adam optimizer
                .list()
                .layer(0, new DenseLayer.Builder().nIn(stateSize).nOut(hiddenDim)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(hiddenDim).nOut(hiddenDim)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE) // Use MSE loss for Q-learning regression
                        .nIn(hiddenDim).nOut(actionSize)
                        .activation(Activation.IDENTITY) // Linear output for Q-values
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init(); // Initialize parameters
        return model;
    }
}