# Gaussian Process Explanation Uncertainty (GPEC)

Implementation of GPEC, a method of estimating uncertainty for explanations that can capture the decision boundary complexity of the black-box model.

**UNDER CONSTRUCTION**: Will be updated by AISTATS 2024 (May 2024)

<br />

![Image](https://github.com/davinhill/GPEC/raw/main/Figures/fig1.jpg)

For more details, please see the full manuscript:

**Boundary-Aware Uncertainty for Feature Attribution Explainers**  
Davin Hill, Aria Masoomi, Max Torop, Sandesh Ghimire, Jennifer Dy  
*Proceedings of the 27th International Conference on Artificial Intelligence and Statistics (AISTATS) 2024*  
[[Paper]](https://arxiv.org/abs/2210.02419) [[Poster]](https://drive.google.com/file/d/1heCKUitA9mcXFK61565-a46xcKoZA8nh/view?usp=share_link)

<!-- # Examples
We have an [example implementation](https://github.com/davinhill/BivariateShapley/blob/main/Examples/example1_sentimentanalysis.ipynb) for a toy dataset, which can be run on Google Colab.

# GPEC Visualization Tool

[[Visualizer]](https://gpec-demo.onrender.com) (Note: may take 3-5min to load)

Alternatively, the source code is available in [./Plotly_visualization](https://github.com/davinhill/GPEC/tree/main/Plotly_visualization) and can be run locally using Plotly and Dash.

<img src="https://github.com/davinhill/GPEC/raw/main/Figures/plotly1.png" alt="GPEC Visualization: Decision Boundary" width="70%"/>
<img src="https://github.com/davinhill/GPEC/raw/main/Figures/plotly2.png" alt="GPEC Visualization: WEG Kernel Similarity" width="70%"/>
 -->

# Experiments

Below we detail the code used to evaluate GPEC, as described in the Experiments section of the paper.


**Datasets and Black-Box Models:**
The black-box models evaluated in the experiments section are trained using the code in the [./Models/blackbox_model_training](https://github.com/davinhill/GPEC/tree/main/Tests/Models/blackbox_model_training) folder. Datasets are not included in the repository due to file size, however all datasets are publically available with sources listed in the paper supplement.

**Uncertainty Visualization (Figure 4 and Figure 6):** [a_kernel_test_rev3.py](https://github.com/davinhill/GPEC/blob/main/Tests/uncertaintyfigure/a_kernel_test_rev3.py)

**Regularization Test (Table 1):** [a_regularization_test.py](https://github.com/davinhill/GPEC/blob/main/Tests/regularization_test/a_regularization_test.py)



# Citation
```
@misc{hill2024gpec,
      title={Boundary-Aware Uncertainty for Feature Attribution Explainers}, 
      author={Davin Hill and Aria Masoomi and Max Torop and Sandesh Ghimire and Jennifer Dy},
      year={2024},
      eprint={2210.02419},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
      }
```
