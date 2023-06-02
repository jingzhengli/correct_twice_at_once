#!/usr/bin/env bash
python main_base.py --dataset=clothing1Mbalanced

python main_base.py --dataset=cifar10 --noise_type=symmetric --noise_ratio=20 
python main_base.py --dataset=cifar10 --noise_type=symmetric --noise_ratio=40
python main_base.py --dataset=cifar10 --noise_type=symmetric --noise_ratio=60 
python main_base.py --dataset=cifar10 --noise_type=feature-dependent --noise_ratio=0 
python main_base.py --dataset=cifar10 --noise_type=feature-dependent --noise_ratio=20 
python main_base.py --dataset=cifar10 --noise_type=feature-dependent --noise_ratio=40
python main_base.py --dataset=cifar10 --noise_type=feature-dependent --noise_ratio=60 

python main_base.py --dataset=cifar100 --noise_type=symmetric --noise_ratio=20 
python main_base.py --dataset=cifar100 --noise_type=symmetric --noise_ratio=40 
python main_base.py --dataset=cifar100 --noise_type=symmetric --noise_ratio=60 
python main_base.py --dataset=cifar100 --noise_type=feature-dependent --noise_ratio=0 
python main_base.py --dataset=cifar100 --noise_type=feature-dependent --noise_ratio=20 
python main_base.py --dataset=cifar100 --noise_type=feature-dependent --noise_ratio=40
python main_base.py --dataset=cifar100 --noise_type=feature-dependent --noise_ratio=60 

