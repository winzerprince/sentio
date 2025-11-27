# Model Evaluation

In order to have a better understandanding of which architectures worked based and why, we carried out an evaluation  
on 6 models namely Xgboost, SVR,Ridge Regression Model, CRNN, ViT, AST(Audio Spectogram transformer). We trained each of
the models with the same dataset specifically using static_annotations_averaged_per_song present in the DEAM dataset ( MediaEval Database for Emotional Analysis of Music) for the target variables.The CRNN, Vit and AST used mel spectograms as input while the Xgboost, SVR and Ridge Regression model used a selected set of features from the features folder in the DEAM dataset . We carried out this process starting with the more basic models and our observations infulenced the decisions for the next models to train and how train them.

We got interesting results that will be listed in the table x.x, the primary evaluation metric we used for the Models with Vision transformer architecture was Concordence Correlation Coefficient [1] because it considers performance in both trend and absolute scale thus tracking correlation, mismatch in mean and variance for models with the transformer architecture that can output continous emotion estiamtes. For the more basic models such as SVR and Xgboost we used R^2 as the primary metice since such models can only only demonstrate basic predictions that are independent of contionous analysis of a song The results are detailed in table x.x

citations

  <div class="csl-entry">
    <div class="csl-left-margin">[1]</div><div class="csl-right-inline">J. Han, Z. Zhang, Z. Ren, B. Schuller, and B. Schuller, “Exploring Perception Uncertainty for Emotion Recognition in Dyadic Conversation and Music Listening,” <i>Cognitive Computation</i>, vol. 13, no. 2, pp. 231–240, Mar. 2021, doi: 10.1007/S12559-019-09694-4.</div>
  </div>
