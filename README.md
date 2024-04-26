# Gaussian Process Explanation Uncertainty (GPEC)

Implementation of GPEC, a method of estimating uncertainty for explanations that can capture the decision boundary complexity of the black-box model.

**UNDER CONSTRUCTION**: Will be updated by AISTATS 2024 (May 2024)

<br />

![Image](https://github.com/davinhill/GPEC/raw/main/Figures/fig1.jpg)

For more details, please see our full paper:

**Boundary-Aware Uncertainty for Feature Attribution Explainers**  
Davin Hill, Aria Masoomi, Max Torop, Sandesh Ghimire, Jennifer Dy  
*Proceedings of the 27th International Conference on Artificial Intelligence and Statistics (AISTATS) 2024*  
[[Paper]](https://proceedings.mlr.press/v238/hill24a.html) [[Poster]](https://drive.google.com/file/d/1heCKUitA9mcXFK61565-a46xcKoZA8nh/view?usp=share_link)

<!-- # Examples
We have an [example implementation](https://github.com/davinhill/BivariateShapley/blob/main/Examples/example1_sentimentanalysis.ipynb) for a toy dataset.

To use GPEC with explainer, you need ... packages.

-->

# GPEC Visualization Tool

The GPEC Visualizer is an interactive explanation of GPEC and the WEG Kernel. The tool lets you see how changing model complexity, number of DB samples, and $\lambda$ and $\rho$ values affect the GPEC uncertainty estimate.

[[Visualizer]](https://gpec-demo.onrender.com) (Note: may take 3-5min to load)

Alternatively, the source code is available in the [Plotly_visualization](https://github.com/davinhill/GPEC/tree/main/Plotly_visualization) directory and can be run locally using Plotly and Dash.

<img src="https://github.com/davinhill/GPEC/raw/main/Figures/plotly1.png" alt="GPEC Visualization: Decision Boundary" width="70%"/>
<img src="https://github.com/davinhill/GPEC/raw/main/Figures/plotly2.png" alt="GPEC Visualization: WEG Kernel Similarity" width="70%"/>


# Running GPEC
GPEC_Explainer() in [GPEC.py](https://github.com/davinhill/GPEC/blob/main/GPEC/GPEC.py) is a wrapper that can generate explanations and calculate the GPEC uncertainty estimate. This wrapper has a number of explainers built in (see [Explainers included with GPEC](#explainers-included-with-gpec)) but it can be used with any feature attribution explainer by pre-calculating the explanations for the training set. It will also approximate the black-box decision boundary, as outlined in the manuscript, but it can used with pre-calculated or synthetic decision boundary samples. Saving and loading the decision boundary samples can also save time when using the same black-box model with different explainers or hyperparameters.

Initialize GPEC with:
```
gpec = GPEC_Explainer(f_blackbox, x_train, y_train)
```
f_blackbox is a function that takes data samples as input and returns the model output. x_train is a numpy matrix of the training data. y_train are the training labels, which is required for some explainers.

Once GPEC is trained, it can be used to explain test samples using:
```
gpec.explain(x_test)
```
Descriptions of other GPEC parameters / arguments can be found in [GPEC.py](https://github.com/davinhill/GPEC/blob/main/GPEC/GPEC.py)

## Explainers included with GPEC
The GPEC wrapper can (optionally) generate explanations for the following explainers: KernelSHAP, LIME, BayesSHAP, BayesLIME, CXPlain, Shapley Sampling Values. Note that the correct package(s) need to be pre-installed in order to use respective explanation method.

**KernelSHAP**  
Implementation: https://github.com/shap/shap
```
pip install shap
```

**LIME**  
Implementation: https://github.com/marcotcr/lime
```
pip install lime
```

**CXPlain**  
Implementation: https://github.com/d909b/cxplain
```
pip install cxplain
```



**BayesSHAP and BayesLIME**  
Implementation: https://github.com/dylan-slack/Modeling-Uncertainty-Local-Explainability  

From the root directory of the GPEC repository:
```
cd ..
git clone https://github.com/dylan-slack/Modeling-Uncertainty-Local-Explainability.git
```


**Shapley Sampling Values**  
Implementation: https://github.com/davinhill/BivariateShapley  

From the root directory of the GPEC repository:
```
cd ..
git clone https://github.com/davinhill/BivariateShapley.git
```






# Experiments

Below we detail the code used to evaluate GPEC, as described in the Experiments section of the paper.


**Datasets and Black-Box Models:**
The black-box models evaluated in the experiments section are trained using the code in the [Models/blackbox_model_training](https://github.com/davinhill/GPEC/tree/main/Tests/Models/blackbox_model_training) directory. Datasets are not included in the repository due to file size, however all datasets are publically available with sources listed in the paper supplement.

**Uncertainty Visualization (Figure 4 and Figure 6):** [a_kernel_test_rev3.py](https://github.com/davinhill/GPEC/blob/main/Tests/uncertaintyfigure/a_kernel_test_rev3.py)

**Regularization Test (Table 1):** [a_regularization_test.py](https://github.com/davinhill/GPEC/blob/main/Tests/regularization_test/a_regularization_test.py)



# Citation
```
@InProceedings{hill2024gpec,
  title = {Boundary-Aware Uncertainty for Feature Attribution Explainers },
  author = {Hill, Davin and Masoomi, Aria and Torop, Max and Ghimire, Sandesh and Dy, Jennifer},
  booktitle = {Proceedings of The 27th International Conference on Artificial Intelligence and Statistics},
  pages = {55--63},
  year = {2024},
  editor = {Dasgupta, Sanjoy and Mandt, Stephan and Li, Yingzhen},
  volume = {238},
  series = {Proceedings of Machine Learning Research},
  month = {02--04 May},
  publisher = {PMLR},
  pdf = {https://proceedings.mlr.press/v238/hill24a/hill24a.pdf},
  url = {https://proceedings.mlr.press/v238/hill24a.html},
}
```
