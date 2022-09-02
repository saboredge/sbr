import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sbr.layers import BADBlock

def one_layer_multicategorical(input_size = None,
                               output_size = None,
                               learning_rate: float = 0.0001,
                               dim: int = 1000,
                               specificityAtSensitivityThreshold: float = 0.50,
                               sensitivityAtSpecificityThreshold: float = 0.50,
                               kernel_initializer = tf.keras.initializers.HeNormal(), 
                               bias_initializer = tf.zeros_initializer(),
                               output_activation: str ='softmax',
                               isMultilabel: bool = True,
                               seed = None,
                               verbose: bool = True):
    """Compile a single layer multicategorical model. 

    Can use `sbr.visualize.plot_loss_curve` to see the metrics after fitting

    Args:
      input_size: Usually `x_train.shape[1]`; not required for compile, but for calling `model.summary()`

      output_size: Number of classes in the one-hot-encoded target vector; usually `y_train.shape[1]`

      learning_rate: Plan for this to be reduced during `EarlyStopping` checkpoints in the model training/fit

      dim: Number of nodes to have in the hidden layer. Somthing half-way between input_size and output_size is a good choice, but if input_size is very big, the number may need to be smaller in order to reduce the number of trainable parameters and avoid over-fitting.

      specificityAtSensitivityThreshold: With this percentage of sensitivity (e.g., detecting at least this many true positives), find the specificity (e.g., how many identified will actually be correct). This is a bit trickier for multivariate problems, see `this blog artical on analyticsvidhya.com <https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/>`_

      sensitivityAtSpecificityThreshold: Same as above, but for specificity.

      kernel_initializer: HeNormal initializer forces diversity of outcomes between trainings

      bias_initializder: initialize biases

      output_activation: Use `softmax` for multicategorical, one-hot encoded

      isMultilabel: Should alwasy be True for multicategorical models
    
      seed: used with tensorflow, numpy, python to make random number generator create reproducible results. Also use 'seed' in initializer,

      verbose: If True, print model summary. Set to False if input_size = None to avoid error

    Returns:
      model of type 
      `tf.keras.model <https://keras.io/api/models/model#model-class>`_





    Example usage:

      >>> model = compile.one_layer_multicategorical(input_size=x_train.shape[1],
                                           output_size=y_train.shape[1],
                                           output_activation='softmax',
                                           learning_rate=0.0001,
                                           isMultilabel=True,
                                           dim=1000,
                                           specificityAtSensitivityThreshold=0.50,
                                           sensitivityAtSpecificityThreshold=0.50,
                                           seed = 42,
                                           verbose=True)
        Model: "sequential"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        Input_BAD (BADBlock)         (None, 1000)              18968000  
        _________________________________________________________________
        output (Dense)               (None, 26)                26026     
        =================================================================
        Total params: 18,994,026
        Trainable params: 18,992,026
        Non-trainable params: 2,000
        _________________________________________________________________

    """
    import tensorflow as tf
    import  numpy as np
    
    if input_size == None and verbose:
        print("! Turning off verbose: model.summary() will cause an error if model is compiled with unknown input size.")
        verbose = False
    # python random seed not used
    # import random
    # random.seed(seed)
    # tensorflow seed must be set
    tf.random.set_seed(seed)
    # xxx see if this is required?: - doesn't seem like graph needs to be reset either
    # from tensorflow.python.framework import ops
    # ops.reset_default_graph()

    # numpy random seed not used
    # np.random.seed(seed)

    model = Sequential()
    model.add(BADBlock(dim, input_dim=input_size, name="Input_BAD",
                       activation='relu',
                       dropout_rate = 0.50,
                       kernel_initializer=kernel_initializer,
                       bias_initializer=bias_initializer
                    ))
    model.add(Dense(output_size, activation=output_activation, name = "output",
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer
                    ))

    model.compile(loss='categorical_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(learning_rate),
                  metrics=['accuracy', 'mse',
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.AUC(name='auc',
                                                multi_label=isMultilabel),
                           tf.keras.metrics.SpecificityAtSensitivity(specificityAtSensitivityThreshold, name='SpecificityAtSensitivity'),
                           tf.keras.metrics.SensitivityAtSpecificity(sensitivityAtSpecificityThreshold, name='SensitivityAtSpecificity'),
                           tf.keras.metrics.FalsePositives(name='fp'),
                           tf.keras.metrics.FalseNegatives(name='fn'),
                           tf.keras.metrics.TruePositives(name='tp'),
                           tf.keras.metrics.TrueNegatives(name='tn')
                       ])
    if verbose:
        model.summary()
    return model
