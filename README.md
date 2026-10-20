# Evaluating an Evaluation: Membership Inference Attacks as Machine Unlearning Diagnostics

**Umid Suleymanov, Laman Aliyeva, Nihat Abdullayev, Saida Zarbiyeva, Murat Kantarcioglu**

*NeurIPS 2026, Evaluations & Datasets Track*

This repository contains the code, base-model training logs and per-seed results of the paper. It covers image classification on CIFAR-10, CIFAR-20, MNIST and MUCAC with AllCNN, ResNet-18 and ViT, and LLM unlearning on TOFU with Phi-1.5. Every experiment is evaluated with membership inference attacks (MIA) and the MIAU score.

## Repository structure

    code/           experiment, metric, figure and table code
    loss/           training logs of the base (original) models
    data/           per-seed results with MIAU scores, one CSV per experiment

## code/

### Running experiments

| File | Purpose |
|---|---|
| `run_all_experiments.ipynb` | Runs every image-classification experiment (Google Colab, GPU). Configured by `config.json`. |
| `run_tofu_experiment.ipynb` + `tofu_retrain_experiment.py` | TOFU experiment on Phi-1.5 with true retraining and DualOptim+ |
| `train_model.py` | Trains a baseline (original) model |
| `main.py` → `forget.py` | Standard experiments (entropy-based MIA) |
| `main_new_metrics.py` → `forget_new_metrics.py` | Advanced attacks: XGBoost on saliency maps, LiRA, Quantile and Shadow (`metrics_mia.py`, `lira/`, `quantile/`, `shadow_attack/`) |
| `main_dualoptim.py` → `forget_dualoptim.py`, `dualoptim_adapter.py` | SCRUB, SalUn and SFR-on, with and without DualOptim (`dualoptim/` is the unmodified upstream code) |
| `forget_full_class_main.py`, `forget_subclass_main.py` | Full-class and sub-class forgetting on CIFAR-20 |
| `generate_splits.py`, `generate_mucac_split.py`, `splits/` | Forget/retain splits used in the paper |
| `download_mucac.py` | Downloads the MUCAC data |

### Results and MIAU

| File | Purpose |
|---|---|
| `convert_csv.py` | Compiles per-run outputs into one CSV per experiment |
| `calculate_miau_per_seed.py` | Adds the f_i, MUS_i and MIAU columns to a compiled CSV |
| `postprocess.py` | One-time corrections and the ablation CSVs. These are already applied to `data/`; do not run it again on those files. |

### Figures and tables

`mia_critique_visualizations.py`, `advanced_mia_visualizations.py`, `bargraph_retrains.py`, `box_plots_baseline_retrain.py`, `p_value_heatmap.py`, `effect_sizes.py`, `sensitivity_analysis.py`, `generate_llm_visuals.py`, `train_test_loss_visualization.py`, `generate_latex_tables_neurips.py`

### Requirements

Python 3 with PyTorch and torchvision, plus

    pip install numpy pandas scipy scikit-learn xgboost matplotlib seaborn tqdm torchmetrics

The ViT experiments need `transformers==4.30.0` and `tokenizers==0.13.3` (the ViT backbone is `google/vit-base-patch16-224`). The TOFU experiment needs `transformers accelerate datasets`. The notebooks install everything in their first cells.

### Paths

- **Notebooks:** both expect the project on Google Drive. Set `BASE_PATH` in `run_all_experiments.ipynb` (and `base_path` in `config.json`) to the folder containing the code, and `NOTEBOOK_FOLDER` in `run_tofu_experiment.ipynb`.
- **Datasets:** datasets are not included in this repository. The experiment scripts read them from `./data`, relative to the working directory, which is `code/data` when they run from `code/`. This is a separate folder from the repository's top-level `data/`.
  - CIFAR-10, CIFAR-20 (the CIFAR-100 superclasses) and MNIST are downloaded there automatically by torchvision.
  - MUCAC is downloaded and extracted by `download_mucac.py` into `code/data/MUCAC2`. Rename that folder to `code/data/MUCAC`. The scripts expect `CelebA-HQ-identity.txt`, `CelebA-HQ-attribute.txt` and `CelebAMask-HQ/CelebA-HQ-img/` inside it.
  - TOFU is loaded from Hugging Face `datasets` (`locuslab/TOFU`).
- **Figure and table scripts:** these set `RESULTS_ROOT`, `OUTPUT_DIR` or the input file paths at the top of each file. Point them at `data/` and `loss/`. They read `Results_<Experiment>/compiled_results_MIAU.csv`, which is `data/<Experiment>_MIAU.csv` in this repository.
- **Checkpoints:** model checkpoints are not included. Baseline models are trained with `train_model.py`, and retrained models are produced by the `retrain` method.

## loss/

Training logs of the original models, one file per model and dataset. Each epoch has two lines:

    Epoch 1 - Average Train Loss: 1.6840, Train Accuracy: 0.3854
    Test set: Epoch: 1, Average loss: 0.0065, Accuracy: 0.4620, Time consumed:3.49s

| File | Model |
|---|---|
| `allcnn_cifar10.txt`, `resnet_cifar10.txt`, `vit_cifar10.txt` | AllCNN / ResNet-18 / ViT on CIFAR-10 |
| `allcnn_cifar20.txt`, `resnet_cifar20.txt` | AllCNN / ResNet-18 on CIFAR-20 |
| `allcnn_mnist.txt`, `resnet_mnist.txt` | AllCNN / ResNet-18 on MNIST |
| `resnet_mucac.txt` | ResNet-18 on MUCAC |
| `ResNet18-Cifar10-Overfitted-best.txt` | ResNet-18 on CIFAR-10 without augmentation (overfitted) |
| `ResNet18-Cifar10-Underfitted-best.txt` | ResNet-18 on CIFAR-10 trained for 1 epoch (underfitted) |

`code/train_test_loss_visualization.py` plots these logs.

## data/

### Image classification

Each experiment has two files:

- `<Experiment>_MIAU.csv`: MIAU over all three MIA tasks (Forget vs Retain, Forget vs Test, Test vs Retain)
- `<Experiment>_ablation_MIAU.csv`: MIAU recomputed without the Test vs Retain task

Each row is one unlearning method run on one seed.

| Experiment | Setting | Methods × seeds |
|---|---|---|
| `AllCNN_Cifar10`, `ResNet18_Cifar10`, `ViT_Cifar10` | CIFAR-10, random forgetting | 15 × 10 |
| `AllCNN_Cifar20`, `ResNet18_Cifar20` | CIFAR-20, random forgetting | 15 × 10 |
| `AllCNN_Mnist`, `ResNet18_Mnist` | MNIST, random forgetting | 15 × 10 |
| `ResNet18_MUCAC` | MUCAC, identity-based forgetting | 15 × 10 |
| `AllCNN_Cifar20_FullClass` | CIFAR-20, forget class `electrical_devices` | 9 × 10 |
| `AllCNN_Cifar20_SubClass` | CIFAR-20, forget sub-class `veg` | 9 × 10 |
| `ResNet18_Cifar10_Overfitted` | CIFAR-10, no augmentation | 9 × 10 |
| `ResNet18_Cifar10_Underfitted` | CIFAR-10, 1 training epoch | 9 × 10 |
| `ResNet18_Cifar10_AdvancedMIA_Xgboost` | XGBoost attack on saliency maps | 9 × 10 |
| `ResNet18_Cifar10_AdvancedMIA_Lira`, `_Quantile`, `_Shadow` | LiRA, Quantile and Shadow attacks | 9 × 3 |

**Methods** (`unlearning` column):

- In every experiment: `baseline`, `retrain`, `retrain25`, `retrain50`, `retrain75`, `finetune`, `teacher` (Bad Teaching), `amnesiac` and `ssd`.
- In the 15-method experiments, also `scrub`, `salun`, `sfron` and their DualOptim variants `scrub_do`, `salun_do`, `sfron_do`.

`retrain25/50/75` are models retrained with 25/50/75 % of the forget set kept in training.

**Columns:**

- Run identification: `Filename`, `unlearning`, `dataset`, `model`, `seed`
- Utility: `Test Accuracy`, `Retain Accuracy`, `Forget Set Accuracy (Df)`, `Zero-Retain Forget (ZRF)`, `Method Execution Time`
- Raw MIA scores: `Membership Inference Attack (MIA)`, `Forget vs Retain …`, `Forget vs Test …`, `Test vs Retain …` and `Train vs Test Membership Inference Attack (MIA)`
- Derived scores: `f_i (…)` and `MUS_i (…)` for each MIA task, and the final `MIAU`

### TOFU

`TOFU_MIAU.csv` holds the Phi-1.5 results: 6 methods × 5 seeds, where each seed uses a disjoint forget set of 400 questions.

- Methods: `baseline`, `retrain_25`, `retrain_50`, `retrain_75`, `retrain_100` (full retraining without the forget set) and `dualoptim` (DualOptim+).
- Mean losses: `forget_loss_mean`, `retain_loss_mean`, `test_loss_mean`.
- MIA accuracy and AUC per task: `mia_fvr` / `auc_fvr` (Forget vs Retain), `mia_fvt` / `auc_fvt` (Forget vs Test), `mia_tvr` / `auc_tvr` (Test vs Retain).
- Final score: `miau`.

## Citation

If you use this code or these results, please cite:

    @inproceedings{suleymanov2026evaluating,
      title     = {Evaluating an Evaluation: Membership Inference Attacks as Machine Unlearning Diagnostics},
      author    = {Suleymanov, Umid and Aliyeva, Laman and Abdullayev, Nihat and Zarbiyeva, Saida and Kantarcioglu, Murat},
      booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
      year      = {2026}
    }

## Acknowledgements and references

This code builds on the repositories below. Please also cite the original works when you use the corresponding parts. The BibTeX entries are given at the end of this section.

### Code bases

- Selective Synaptic Dampening, the base of this code: [github.com/if-loops/selective-synaptic-dampening](https://github.com/if-loops/selective-synaptic-dampening)
- Bad Teaching and the ZRF metric: [github.com/vikram2000b/bad-teaching-unlearning](https://github.com/vikram2000b/bad-teaching-unlearning)
- UNSIR: [github.com/vikram2000b/Fast-Machine-Unlearning](https://github.com/vikram2000b/Fast-Machine-Unlearning)
- Fisher forgetting: [github.com/AdityaGolatkar/SelectiveForgetting](https://github.com/AdityaGolatkar/SelectiveForgetting)
- DualOptim (SCRUB, SalUn, SFR-on): [github.com/CityU-MLO/DualOptim](https://github.com/CityU-MLO/DualOptim)
- DualOptim+, used for TOFU: [github.com/CityU-MLO/DualOptimPlus](https://github.com/CityU-MLO/DualOptimPlus)
- Model and training code: [github.com/weiaicunzai/pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100)
- Importance computation in `ssd.py`: [github.com/ContinualAI/avalanche](https://github.com/ContinualAI/avalanche)

### Unlearning methods

- SSD (`ssd`): Jack Foster, Stefan Schoepf, Alexandra Brintrup. *Fast Machine Unlearning without Retraining through Selective Synaptic Dampening.* Proceedings of the AAAI Conference on Artificial Intelligence, 38(11), 12043–12051, 2024. [doi:10.1609/aaai.v38i11.29092](https://doi.org/10.1609/aaai.v38i11.29092)
- Bad Teaching (`teacher`) and ZRF: Vikram S. Chundawat, Ayush K. Tarun, Murari Mandal, Mohan Kankanhalli. *Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks Using an Incompetent Teacher.* Proceedings of the AAAI Conference on Artificial Intelligence, 37(6), 7210–7217, 2023. [doi:10.1609/aaai.v37i6.25879](https://doi.org/10.1609/aaai.v37i6.25879)
- UNSIR: Ayush K. Tarun, Vikram S. Chundawat, Murari Mandal, Mohan Kankanhalli. *Fast Yet Effective Machine Unlearning.* IEEE Transactions on Neural Networks and Learning Systems, 35(9), 13046–13055, 2024. [doi:10.1109/TNNLS.2023.3266233](https://doi.org/10.1109/TNNLS.2023.3266233)
- Amnesiac (`amnesiac`): Laura Graves, Vineel Nagisetty, Vijay Ganesh. *Amnesiac Machine Learning.* Proceedings of the AAAI Conference on Artificial Intelligence, 35(13), 11516–11524, 2021. [doi:10.1609/aaai.v35i13.17371](https://doi.org/10.1609/aaai.v35i13.17371)
- Fisher forgetting: Aditya Golatkar, Alessandro Achille, Stefano Soatto. *Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks.* CVPR, 9304–9312, 2020.
- SCRUB (`scrub`): Meghdad Kurmanji, Peter Triantafillou, Jamie Hayes, Eleni Triantafillou. *Towards Unbounded Machine Unlearning.* NeurIPS 36, 2023.
- SalUn (`salun`): Chongyu Fan, Jiancheng Liu, Yihua Zhang, Eric Wong, Dennis Wei, Sijia Liu. *SalUn: Empowering Machine Unlearning via Gradient-Based Weight Saliency in Both Image Classification and Generation.* ICLR, 2024.
- SFR-on (`sfron`): Zhehao Huang, Xinwen Cheng, JingHao Zheng, Haoran Wang, Zhengbao He, Tao Li, Xiaolin Huang. *Unified Gradient-Based Machine Unlearning with Remain Geometry Enhancement.* NeurIPS 37, 2024. [doi:10.52202/079017-0831](https://doi.org/10.52202/079017-0831)
- DualOptim (`*_do`): Xuyang Zhong, Haochen Luo, Chen Liu. *DualOptim: Enhancing Efficacy and Stability in Machine Unlearning with Dual Optimizers.* NeurIPS 38, 2025.
- DualOptim+ (TOFU `dualoptim`): Xuyang Zhong, Qizhang Li, Steven Y. Guo, Chen Liu. *DualOptim+: Bridging Shared and Decoupled Optimizer States for Better Machine Unlearning in Large Language Models.* ICML, PMLR 306, 164420–164438, 2026.

### Membership inference attacks

- Shadow attack: Reza Shokri, Marco Stronati, Congzheng Song, Vitaly Shmatikov. *Membership Inference Attacks Against Machine Learning Models.* IEEE Symposium on Security and Privacy (SP), 3–18, 2017. [doi:10.1109/SP.2017.41](https://doi.org/10.1109/SP.2017.41)
- LiRA: Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramèr. *Membership Inference Attacks From First Principles.* IEEE Symposium on Security and Privacy (SP), 1897–1914, 2022.
- Quantile attack: Martin Bertran, Shuai Tang, Michael Kearns, Jamie Morgenstern, Aaron Roth, Zhiwei Steven Wu. *Scalable Membership Inference Attacks via Quantile Regression.* NeurIPS 36, 2023.
- MIA on model explanations (the saliency-map attack): Reza Shokri, Martin Strobel, Yair Zick. *On the Privacy Risks of Model Explanations.* AIES, 231–241, 2021. [doi:10.1145/3461702.3462533](https://doi.org/10.1145/3461702.3462533); and Catherine Huang, Martin Pawelczyk, Himabindu Lakkaraju. *Explaining the Model, Protecting Your Data: Revealing and Mitigating the Data Privacy Risks of Post-Hoc Model Explanations via Membership Inference.* ICML 2024 Workshop on Next Generation of AI Safety, 2024.
- XGBoost: Tianqi Chen, Carlos Guestrin. *XGBoost: A Scalable Tree Boosting System.* KDD, 785–794, 2016. [doi:10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)

### Datasets and models

- CIFAR-10 and CIFAR-100/20: Alex Krizhevsky, Geoffrey Hinton. *Learning Multiple Layers of Features from Tiny Images.* Technical report, University of Toronto, 2009.
- MNIST: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner. *Gradient-Based Learning Applied to Document Recognition.* Proceedings of the IEEE, 86(11), 2278–2324, 1998. [doi:10.1109/5.726791](https://doi.org/10.1109/5.726791)
- MUCAC: Dasol Choi, Dongbin Na. *Towards Machine Unlearning Benchmarks: Forgetting the Personal Identities in Facial Recognition Systems.* 2023. [arXiv:2311.02240](https://arxiv.org/abs/2311.02240)
- TOFU: Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary C. Lipton, J. Zico Kolter. *TOFU: A Task of Fictitious Unlearning for LLMs.* First Conference on Language Modeling (COLM), 2024.
- Phi-1.5: Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, Yin Tat Lee. *Textbooks Are All You Need II: phi-1.5 Technical Report.* 2023. [arXiv:2309.05463](https://arxiv.org/abs/2309.05463)
- ResNet: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *Deep Residual Learning for Image Recognition.* CVPR, 770–778, 2016.
- AllCNN: Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, Martin Riedmiller. *Striving for Simplicity: The All Convolutional Net.* ICLR Workshop Track, 2015.
- ViT: Alexey Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR, 2021.
- ViT checkpoint: Google. *google/vit-base-patch16-224* (Hugging Face model card, Apache-2.0 license). [huggingface.co/google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)
- ImageNet: Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, Li Fei-Fei. *ImageNet: A Large-Scale Hierarchical Image Database.* CVPR, 248–255, 2009. [doi:10.1109/CVPR.2009.5206848](https://doi.org/10.1109/CVPR.2009.5206848)

### Software

PyTorch, TorchVision, Hugging Face Transformers, Datasets and Accelerate, scikit-learn, XGBoost, NumPy, pandas, SciPy, Matplotlib, seaborn and TorchMetrics.

### BibTeX

<details>
<summary>Show BibTeX entries</summary>

```bibtex
@inproceedings{Foster2024AAAI,
  author    = {Jack Foster and Stefan Schoepf and Alexandra Brintrup},
  title     = {Fast Machine Unlearning without Retraining through Selective Synaptic Dampening},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {38},
  number    = {11},
  pages     = {12043--12051},
  year      = {2024},
  doi       = {10.1609/aaai.v38i11.29092}
}

@inproceedings{Chundawat2023AAAI,
  author    = {Vikram S. Chundawat and Ayush K. Tarun and Murari Mandal and Mohan Kankanhalli},
  title     = {Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks Using an Incompetent Teacher},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {37},
  number    = {6},
  pages     = {7210--7217},
  year      = {2023},
  doi       = {10.1609/aaai.v37i6.25879}
}

@article{Tarun2023TNNLS,
  author    = {Ayush K. Tarun and Vikram S. Chundawat and Murari Mandal and Mohan Kankanhalli},
  title     = {Fast Yet Effective Machine Unlearning},
  journal   = {IEEE Transactions on Neural Networks and Learning Systems},
  volume    = {35},
  number    = {9},
  pages     = {13046--13055},
  year      = {2024},
  doi       = {10.1109/TNNLS.2023.3266233}
}

@inproceedings{Graves2021AAAI,
  author    = {Laura Graves and Vineel Nagisetty and Vijay Ganesh},
  title     = {Amnesiac Machine Learning},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {35},
  number    = {13},
  pages     = {11516--11524},
  year      = {2021},
  doi       = {10.1609/aaai.v35i13.17371}
}

@inproceedings{golatkar2020sunshine,
  author    = {Aditya Golatkar and Alessandro Achille and Stefano Soatto},
  title     = {Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {9304--9312},
  year      = {2020}
}

@inproceedings{Kurmanji2023,
  author    = {Meghdad Kurmanji and Peter Triantafillou and Jamie Hayes and Eleni Triantafillou},
  title     = {Towards Unbounded Machine Unlearning},
  booktitle = {Advances in Neural Information Processing Systems 36 (NeurIPS 2023)},
  year      = {2023}
}

@inproceedings{Fan2024SalUn,
  author    = {Chongyu Fan and Jiancheng Liu and Yihua Zhang and Eric Wong and Dennis Wei and Sijia Liu},
  title     = {{SalUn}: Empowering Machine Unlearning via Gradient-Based Weight Saliency in Both Image Classification and Generation},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024}
}

@inproceedings{Huang2024SFRon,
  author    = {Zhehao Huang and Xinwen Cheng and JingHao Zheng and Haoran Wang and Zhengbao He and Tao Li and Xiaolin Huang},
  title     = {Unified Gradient-Based Machine Unlearning with Remain Geometry Enhancement},
  booktitle = {Advances in Neural Information Processing Systems 37 (NeurIPS 2024)},
  year      = {2024},
  doi       = {10.52202/079017-0831}
}

@inproceedings{zhong2025dualoptim,
  author    = {Xuyang Zhong and Haochen Luo and Chen Liu},
  title     = {{DualOptim}: Enhancing Efficacy and Stability in Machine Unlearning with Dual Optimizers},
  booktitle = {Advances in Neural Information Processing Systems 38 (NeurIPS 2025)},
  year      = {2025}
}

@inproceedings{zhong2026dualoptimplus,
  author    = {Xuyang Zhong and Qizhang Li and Steven Y. Guo and Chen Liu},
  title     = {{DualOptim+}: Bridging Shared and Decoupled Optimizer States for Better Machine Unlearning in Large Language Models},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  pages     = {164420--164438},
  year      = {2026},
  publisher = {PMLR}
}

@inproceedings{shokri2017membership,
  author    = {Reza Shokri and Marco Stronati and Congzheng Song and Vitaly Shmatikov},
  title     = {Membership Inference Attacks Against Machine Learning Models},
  booktitle = {2017 IEEE Symposium on Security and Privacy (SP)},
  pages     = {3--18},
  year      = {2017},
  publisher = {IEEE},
  doi       = {10.1109/SP.2017.41}
}

@inproceedings{Carlini2022LiRA,
  author    = {Nicholas Carlini and Steve Chien and Milad Nasr and Shuang Song and Andreas Terzis and Florian Tram{\`e}r},
  title     = {Membership Inference Attacks From First Principles},
  booktitle = {2022 IEEE Symposium on Security and Privacy (SP)},
  pages     = {1897--1914},
  year      = {2022}
}

@inproceedings{Bertran2023Quantile,
  author    = {Martin Bertran and Shuai Tang and Michael Kearns and Jamie Morgenstern and Aaron Roth and Zhiwei Steven Wu},
  title     = {Scalable Membership Inference Attacks via Quantile Regression},
  booktitle = {Advances in Neural Information Processing Systems 36 (NeurIPS 2023)},
  year      = {2023}
}

@inproceedings{shokri2021privacy,
  author    = {Reza Shokri and Martin Strobel and Yair Zick},
  title     = {On the Privacy Risks of Model Explanations},
  booktitle = {Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society (AIES)},
  pages     = {231--241},
  year      = {2021},
  doi       = {10.1145/3461702.3462533}
}

@inproceedings{huang2024explaining,
  author    = {Catherine Huang and Martin Pawelczyk and Himabindu Lakkaraju},
  title     = {Explaining the Model, Protecting Your Data: Revealing and Mitigating the Data Privacy Risks of Post-Hoc Model Explanations via Membership Inference},
  booktitle = {ICML 2024 Workshop on Next Generation of AI Safety},
  year      = {2024}
}

@inproceedings{chen2016xgboost,
  author    = {Tianqi Chen and Carlos Guestrin},
  title     = {{XGBoost}: A Scalable Tree Boosting System},
  booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)},
  pages     = {785--794},
  year      = {2016},
  doi       = {10.1145/2939672.2939785}
}

@techreport{krizhevsky2009learning,
  author      = {Alex Krizhevsky and Geoffrey Hinton},
  title       = {Learning Multiple Layers of Features from Tiny Images},
  institution = {University of Toronto},
  year        = {2009}
}

@article{lecun1998gradient,
  author    = {Yann LeCun and L{\'e}on Bottou and Yoshua Bengio and Patrick Haffner},
  title     = {Gradient-Based Learning Applied to Document Recognition},
  journal   = {Proceedings of the IEEE},
  volume    = {86},
  number    = {11},
  pages     = {2278--2324},
  year      = {1998},
  doi       = {10.1109/5.726791}
}

@article{choi2023machine,
  author        = {Dasol Choi and Dongbin Na},
  title         = {Towards Machine Unlearning Benchmarks: Forgetting the Personal Identities in Facial Recognition Systems},
  year          = {2023},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  eprint        = {2311.02240}
}

@inproceedings{maini2024tofu,
  author    = {Pratyush Maini and Zhili Feng and Avi Schwarzschild and Zachary C. Lipton and J. Zico Kolter},
  title     = {{TOFU}: A Task of Fictitious Unlearning for {LLMs}},
  booktitle = {First Conference on Language Modeling (COLM)},
  year      = {2024}
}

@misc{li2023textbooks,
  author        = {Yuanzhi Li and S{\'e}bastien Bubeck and Ronen Eldan and Allie Del Giorno and Suriya Gunasekar and Yin Tat Lee},
  title         = {Textbooks Are All You Need {II}: phi-1.5 Technical Report},
  year          = {2023},
  eprint        = {2309.05463},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2309.05463}
}

@inproceedings{he2016deep,
  author    = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
  title     = {Deep Residual Learning for Image Recognition},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {770--778},
  year      = {2016}
}

@inproceedings{springenberg2015allcnn,
  author    = {Jost Tobias Springenberg and Alexey Dosovitskiy and Thomas Brox and Martin Riedmiller},
  title     = {Striving for Simplicity: The All Convolutional Net},
  booktitle = {International Conference on Learning Representations (ICLR), Workshop Track},
  year      = {2015}
}

@inproceedings{dosovitskiy2021vit,
  author    = {Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
  title     = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021}
}

@misc{google2021vitbase,
  author       = {{Google}},
  title        = {{google/vit-base-patch16-224}: Vision Transformer (base-sized model, patch size 16, resolution 224), Apache-2.0 license},
  howpublished = {\url{https://huggingface.co/google/vit-base-patch16-224}},
  note         = {Hugging Face model card}
}

@inproceedings{deng2009imagenet,
  author    = {Jia Deng and Wei Dong and Richard Socher and Li-Jia Li and Kai Li and Li Fei-Fei},
  title     = {{ImageNet}: A Large-Scale Hierarchical Image Database},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {248--255},
  year      = {2009},
  doi       = {10.1109/CVPR.2009.5206848}
}

@inproceedings{paszke2019pytorch,
  author    = {Adam Paszke and Sam Gross and Francisco Massa and Adam Lerer and James Bradbury and Gregory Chanan and Trevor Killeen and Zeming Lin and Natalia Gimelshein and Luca Antiga and Alban Desmaison and Andreas K{\"o}pf and Edward Yang and Zachary DeVito and Martin Raison and Alykhan Tejani and Sasank Chilamkurthy and Benoit Steiner and Lu Fang and Junjie Bai and Soumith Chintala},
  title     = {{PyTorch}: An Imperative Style, High-Performance Deep Learning Library},
  booktitle = {Advances in Neural Information Processing Systems 32 (NeurIPS 2019)},
  pages     = {8024--8035},
  year      = {2019}
}

@misc{torchvision2016,
  author       = {{TorchVision maintainers and contributors}},
  title        = {{TorchVision}: {PyTorch}'s {C}omputer {V}ision Library},
  howpublished = {\url{https://github.com/pytorch/vision}},
  year         = {2016}
}

@inproceedings{wolf2020transformers,
  author    = {Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and R{\'e}mi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander Rush},
  title     = {Transformers: State-of-the-Art Natural Language Processing},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  pages     = {38--45},
  year      = {2020},
  doi       = {10.18653/v1/2020.emnlp-demos.6}
}

@inproceedings{lhoest2021datasets,
  author    = {Quentin Lhoest and Albert Villanova del Moral and Yacine Jernite and Abhishek Thakur and Patrick von Platen and Suraj Patil and Julien Chaumond and Mariama Drame and Julien Plu and Lewis Tunstall and Joe Davison and Mario {\v{S}}a{\v{s}}ko and Gunjan Chhablani and Bhavitvya Malik and Simon Brandeis and Teven Le Scao and Victor Sanh and Canwen Xu and Nicolas Patry and Angelina McMillan-Major and Philipp Schmid and Sylvain Gugger and Cl{\'e}ment Delangue and Th{\'e}o Matussi{\`e}re and Lysandre Debut and Stas Bekman and Pierric Cistac and Thibault Goehringer and Victor Mustar and Fran{\c{c}}ois Lagunas and Alexander Rush and Thomas Wolf},
  title     = {Datasets: A Community Library for Natural Language Processing},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  pages     = {175--184},
  publisher = {Association for Computational Linguistics},
  year      = {2021},
  doi       = {10.18653/v1/2021.emnlp-demo.21}
}

@misc{accelerate2022,
  author       = {Sylvain Gugger and Lysandre Debut and Thomas Wolf and Philipp Schmid and Zachary Mueller and Sourab Mangrulkar and Marc Sun and Benjamin Bossan},
  title        = {Accelerate: Training and inference at scale made simple, efficient and adaptable},
  howpublished = {\url{https://github.com/huggingface/accelerate}}
}

@article{pedregosa2011scikit,
  author    = {Fabian Pedregosa and Ga{\"e}l Varoquaux and Alexandre Gramfort and Vincent Michel and Bertrand Thirion and Olivier Grisel and Mathieu Blondel and Peter Prettenhofer and Ron Weiss and Vincent Dubourg and Jake Vanderplas and Alexandre Passos and David Cournapeau and Matthieu Brucher and Matthieu Perrot and {\'E}douard Duchesnay},
  title     = {Scikit-learn: Machine Learning in {P}ython},
  journal   = {Journal of Machine Learning Research},
  volume    = {12},
  number    = {85},
  pages     = {2825--2830},
  year      = {2011}
}

@article{harris2020numpy,
  author    = {Charles R. Harris and K. Jarrod Millman and St{\'e}fan J. van der Walt and Ralf Gommers and Pauli Virtanen and David Cournapeau and Eric Wieser and Julian Taylor and Sebastian Berg and Nathaniel J. Smith and Robert Kern and Matti Picus and Stephan Hoyer and Marten H. van Kerkwijk and Matthew Brett and Allan Haldane and Jaime Fern{\'a}ndez del R{\'i}o and Mark Wiebe and Pearu Peterson and Pierre G{\'e}rard-Marchant and Kevin Sheppard and Tyler Reddy and Warren Weckesser and Hameer Abbasi and Christoph Gohlke and Travis E. Oliphant},
  title     = {Array programming with {NumPy}},
  journal   = {Nature},
  volume    = {585},
  number    = {7825},
  pages     = {357--362},
  year      = {2020},
  doi       = {10.1038/s41586-020-2649-2}
}

@inproceedings{mckinney2010pandas,
  author    = {Wes McKinney},
  title     = {Data Structures for Statistical Computing in {P}ython},
  booktitle = {Proceedings of the 9th {P}ython in Science Conference},
  editor    = {St{\'e}fan van der Walt and Jarrod Millman},
  pages     = {56--61},
  year      = {2010},
  doi       = {10.25080/Majora-92bf1922-00a}
}

@article{virtanen2020scipy,
  author    = {Pauli Virtanen and Ralf Gommers and Travis E. Oliphant and Matt Haberland and Tyler Reddy and David Cournapeau and Evgeni Burovski and Pearu Peterson and Warren Weckesser and Jonathan Bright and St{\'e}fan J. van der Walt and Matthew Brett and others},
  title     = {{SciPy} 1.0: Fundamental Algorithms for Scientific Computing in {P}ython},
  journal   = {Nature Methods},
  volume    = {17},
  number    = {3},
  pages     = {261--272},
  year      = {2020},
  doi       = {10.1038/s41592-019-0686-2}
}

@article{hunter2007matplotlib,
  author    = {John D. Hunter},
  title     = {Matplotlib: A {2D} Graphics Environment},
  journal   = {Computing in Science \& Engineering},
  volume    = {9},
  number    = {3},
  pages     = {90--95},
  year      = {2007},
  doi       = {10.1109/MCSE.2007.55}
}

@article{waskom2021seaborn,
  author    = {Michael L. Waskom},
  title     = {seaborn: statistical data visualization},
  journal   = {Journal of Open Source Software},
  volume    = {6},
  number    = {60},
  pages     = {3021},
  year      = {2021},
  doi       = {10.21105/joss.03021}
}

@article{detlefsen2022torchmetrics,
  author    = {Nicki Skafte Detlefsen and Jiri Borovec and Justus Schock and Ananya Harsh Jha and Teddy Koker and Luca Di Liello and Daniel Stancl and Changsheng Quan and Maxim Grechkin and William Falcon},
  title     = {{TorchMetrics} -- Measuring Reproducibility in {PyTorch}},
  journal   = {Journal of Open Source Software},
  volume    = {7},
  number    = {70},
  pages     = {4101},
  year      = {2022},
  doi       = {10.21105/joss.04101}
}
```

</details>

