<div id="top"></div>
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Mhackiori/CANEDERLI">
    <img src="figures/logo.png" alt="Logo" width="150" height="150">
  </a>

  <h1 align="center">CANEDERLI</h1>

  <p align="center">
    On The Impact of Adversarial Training and Transferability on CAN Intrusion Detection Systems
    <br />
    <a href="https://doi.org/10.1145/3649403.3656486"><strong>Paper Available »</strong></a>
    <br />
    <br />
    <a href="https://www.math.unipd.it/~fmarchio/">Francesco Marchiori</a>
    ·
    <a href="https://www.math.unipd.it/~conti/">Mauro Conti</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary><strong>Table of Contents</strong></summary>
  <ol>
    <li>
      <a href="#abstract">Abstract</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
    </li>
    <li>
      <a href="#models">Models</a>
    </li>
    <li>
      <a href="#reproducibility">Reproducibility</a>
    </li>
  </ol>
</details>

<div id="abstract"></div>

## 🧩 Abstract

>The growing integration of vehicles with external networks has led to a surge in attacks targeting their Controller Area Network (CAN) internal bus. As a countermeasure, various Intrusion Detection Systems (IDSs) have been suggested in the literature to prevent and mitigate these threats. With the increasing volume of data facilitated by the integration of Vehicle-to-Vehicle (V2V) and Vehicle-to-Infrastructure (V2I) communication networks, most of these systems rely on data-driven approaches such as Machine Learning (ML) and Deep Learning (DL) models. However, these systems are susceptible to adversarial evasion attacks. While many researchers have explored this vulnerability, their studies often involve unrealistic assumptions, lack consideration for a realistic threat model, and fail to provide effective solutions. In this paper, we present **CANEDERLI** (**CAN** **E**vasion **D**etection **R**esi**LI**ence), a novel framework for securing CAN-based IDSs. Our system considers a realistic threat model and addresses the impact of adversarial attacks on DL-based detection systems. Our findings highlight strong transferability properties among diverse attack methodologies by considering multiple state-of-the-art attacks and model architectures. We analyze the impact of adversarial training in addressing this threat and propose an adaptive online adversarial training technique outclassing traditional fine-tuning methodologies. By making our framework publicly available, we aid practitioners and researchers in assessing the resilience of IDSs to a varied adversarial landscape.

<p align="right"><a href="#top">(back to top)</a></p>
<div id="citation"></div>

## 🗣️ Citation

Please, cite this work when referring to CANEDERLI.

```
@inproceedings{10.1145/3649403.3656486,
  author = {Marchiori, Francesco and Conti, Mauro},
  title = {CANEDERLI: On The Impact of Adversarial Training and Transferability on CAN Intrusion Detection Systems},
  year = {2024},
  isbn = {9798400706028},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3649403.3656486},
  doi = {10.1145/3649403.3656486},
  booktitle = {Proceedings of the 2024 ACM Workshop on Wireless Security and Machine Learning},
  pages = {8–13},
  numpages = {6},
  keywords = {adversarial attacks, adversarial training, adversarial transferability, controller area network, intrusion detection systems},
  location = {Seoul, Republic of Korea},
  series = {WiseML '24}
}
```

<p align="right"><a href="#top">(back to top)</a></p>
<div id="usage"></div>

## ⚙️ Usage

To train the models, generate the attacks, and evaluate adversarial transferability and adversarial training, start by cloning the repository.

```bash
git clone https://github.com/Mhackiori/CANEDERLI.git
cd CANEDERLI
```

Then, install the required Python packages by running the following command.

```bash
pip install -r requirements.txt
```

<p align="right"><a href="#top">(back to top)</a></p>
<div id="models"></div>

## 🤖 Models

The [utils](https://github.com/Mhackiori/CANEDERLI/tree/main/utils) directory contains several Python files that are referenced in all the scripts for baseline evaluation and attacks. In particular, details on the models architectures can be found in the [`models.py`](https://github.com/Mhackiori/CANEDERLI/blob/main/utils/models.py) script, and functions for training and evaluation can be found in the [`helpers.py`](https://github.com/Mhackiori/CANEDERLI/blob/main/utils/helpers.py) script. Seed, training details and other parameters can be changed in the [`params.py`](https://github.com/Mhackiori/CANEDERLI/blob/main/utils/params.py) file.

<p align="right"><a href="#top">(back to top)</a></p>
<div id="reproducibility"></div>

## 🔁 Reproducibility

The first step is process the dataset. The original dataset can be found [here](https://ocslab.hksecurity.net/Datasets/survival-ids), but we already include part of the pre-processed data in `.csv` format. By running [`preprocessing.py`](https://github.com/Mhackiori/CANEDERLI/blob/main/preprocessing.py), two datasets for each vehicle will be created in the [dataset](https://github.com/Mhackiori/CANEDERLI/tree/main/dataset) folder (one for binary classification, one for multiclass classification). Models can then be trained by running the [`baseline.py`](https://github.com/Mhackiori/CANEDERLI/blob/main/baseline.py) script. The program will automatically use the already generated models in the [models](https://github.com/Mhackiori/CANEDERLI/tree/main/models) folder, but it is possible to retrain them by deleting the `.pth` files in the directory. The script will also evaluate the models in the same dataset (taking into account the train/test split). Once the models are trained, it is possible to generate the attacks by running [`attacks.py`](https://github.com/Mhackiori/CANEDERLI/blob/main/attacks.py). This will create several `.csv` datasets that will be stored in the [attacks](https://github.com/Mhackiori/CANEDERLI/tree/main/attacks) folder (here gitignored due to storage constraints). The [`evaluation.py`](https://github.com/Mhackiori/CANEDERLI/blob/main/evaluation.py) is used for evaluating each model on each adversarial dataset. This will generate the `.csv` files in the [results](https://github.com/Mhackiori/CANEDERLI/tree/main/models) folder. As we used a seed in all our scripts, results should be (almost exactly) the same. We only notice a few discrepancies when training/testing the models on different hardware, i.e., CPU or GPU. As such, if a GPU is accessible when trying to reproduce the results, its usage is recommended. Finally, adverarial training in both fine-tuning and online modes are taken care of in the [`adversarial-training.py`](https://github.com/Mhackiori/CANEDERLI/blob/main/adversarial-training.py) script, which also automatically performs its evaluation (stored in the [results](https://github.com/Mhackiori/CANEDERLI/tree/main/models) folder). The [`results.ipynb`](https://github.com/Mhackiori/CANEDERLI/blob/main/results.ipynb) notebook then organize the results as shown in the paper.

<p align="right"><a href="#top">(back to top)</a></p>
