# Adaptive Attention Span in Computer Vision

Official implementation of [Adaptive Attention Span in Computer Vision](https://arxiv.org/abs/2004.08708).

In this work we first try replicating results from [Stand-Alone Self-Attention in Vision Models](https://arxiv.org/abs/1906.05909).

Next we propose a novel method based on the [Adaptive Attention Span](https://arxiv.org/abs/1905.07799) for learning a local self attention kernel size.
We compare this with [Local Attention](https://arxiv.org/abs/1906.05909) kernels as well as convolution kernels on CIFAR100.
Our codes for Adaptive Attention Span in 2D is originally inspired from [FAIR's implementation](https://github.com/facebookresearch/adaptive-span/blob/master/adaptive_span.py).
Code for self-attention in convolutions is loosely based on [this repo](https://github.com/leaderj1001/Stand-Alone-Self-Attention) by [leaderj1001](https://github.com/leaderj1001).


### Steps to replicate
1. Clone this repository
2. Get the requirements ```pip install -r requirements.txt```

Execution notes:
* Our Adaptive implementation takes 3, 6 and 11 hours for small, medium and large models respectively on 2 P100 GPUs for 100 epochs on CIFAR100.
* Some important flags are,
    * To run on GPU, use the flag ```--cuda True```, otherwise do not use this option.
    * Use flags ```--smallest_version True``` to run the smallest version. ```--small_version True``` to run the medium model and no flags to use the large model
    * A description of each of the small, medium and large is given in Appendix A.3 of our [paper](https://arxiv.org/abs/2004.08708)
* For more details on other flags, see the file config.py which has descriptions for each.

### Snippets
Best performing medium adaptive attention span model on CIFAR100:
```
python main.py --all_attention True --eta_min 0 --warmup_epochs 10 \
--lr 0.05 --batch-size 50 --small_version True --cuda True \
--num-workers 2 --xpid best_adaptive_medium --groups 4 \
--attention_kernel 5 --epochs 100 --dataset CIFAR100 --weight-decay 0.0005 \
--adaptive_span True --R 2 --span_penalty 0.01
```

Best performing medium local attention model on CIFAR100:
```
python main.py --all_attention True --eta_min 0 --warmup_epochs 10 \
--lr 0.05 --batch-size 50 --small_version True --cuda True \
--num-workers 2 --xpid best_local_medium --groups 4 \
--attention_kernel 5 --epochs 100  --dataset CIFAR100 --weight-decay 0.0005
```

Best performing medium CNN model on CIFAR100:
```
python main.py --eta_min 0 --warmup_epochs 10 --lr 0.2 --batch-size 50 \
--small_version True --cuda True --num-workers 2 --T_max 100 --xpid best_cnn_medium \
--dataset CIFAR100 --force_cosine_annealing True --weight-decay 0.0001
```

### Reference
If you find this repository useful, do cite it with 
```
@misc{parker2020adaptive,
    title={Adaptive Attention Span in Computer Vision},
    author={Jerrod Parker and Shakti Kumar and Joe Roussy},
    year={2020},
    eprint={2004.08708},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
