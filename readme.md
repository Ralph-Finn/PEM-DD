## Gift from Modern Nature: Potential Energy Minimization for Explainable Dataset Distillation

@ Wenbin Yang Ralph.Yang@dell.com
@ Zijia Wang zijia_wang@dell.com

Reference: https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration

#### you can test the dataset distillation on miniImageNet data set as
```
python dataset_distillation.py --data_path "./data/data.pkl" --dataset_type "mini"
```

#### you can test the few_shot learning on miniImageNet data set as
```
python few_shot_batch.py --data_path "./data/data.pkl" --dataset_type "mini" --test_num 10
```

#### Moreover, we also provide the Jupyter notebook as a testing playground.

Test_CUB ---> Test the basic tasks in CUB dataset. (dataset fusion, dataset distillation with UT).
FewShot_CUB --> Visulize the fewshot learning and parameter founding in CUB dataset.
Hyper ---> Test the Hyper Paramters of our PEM method.
