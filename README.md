# Unleashing the Tiger: Inference Attacks on Split Learning
We investigate the security of *split learning*---a novel collaborative machine learning framework that enables peak performance by requiring minimal resources consumption. In this paper, we expose novel vulnerabilities of the protocol and demonstrate its inherent insecurity by introducing general attack strategies targeting the reconstruction of clients' private training sets. More prominently, we show that a malicious server can actively hijack the learning process of the distributed model and bring it into an insecure state that enables inference attacks on clients' data. We implement different adaptations of the attack and test them on various datasets as well as within realistic threat scenarios. We demonstrate that our attack is able to overcome recently proposed defensive techniques aimed at enhancing the security of the split learning protocol. Finally, we also illustrate the protocol's insecurity against malicious clients by extending previously devised attacks for Federated Learning.

🎉 **Our paper has been accepted at ACM Conference on Computer and Communications Security 2021 (CCS21)** 🎉 <br>
Paper: [arxiv](https://arxiv.org/abs/2012.02670)

## Code:

This repository contains all the code needed to run the **Feature-space hijacking attack** and its variations. The implementation is based on **TensorFlow2**.

In particular:

*  *FSHA.py*: It implements the attack and a single-user version of split learning. The class *FSHA_binary_property* in the file implements the property-inference attack.
* *architectures.py*: It contains the main network architectures we used in the paper.
* *defense.py*: It implements the distance correlation.
* *datasets.py*: It contains utility to load and parse datasets.

## Proof Of Concepts 🐯:

We report a set of *jupyter notebooks* that act as brief tutorial for the code and replicate the experiments in the paper. Those are:

* *FSHA.ipynb*: It implements the standard Feature-space hijacking attack on the MNIST dataset.
* *FSHA_with_distance_cor.ipynb*: It implements the standard Feature-space hijacking attack on the MNIST dataset, when the distance correlation defense is used from the clients.
* *FSHA_mangled.ipynb*: It implements the Feature-space hijacking attack on the MNIST dataset, when the attacker's training set is mangled of a class.
* ClientSideAttack/clientsideAttack.ipynb: It implements the GAN-based, client-side attack in split learning.
* Work in progress.....

## Cite our work:
```
@inproceedings{10.1145/3460120.3485259,
      author = {Pasquini, Dario and Ateniese, Giuseppe and Bernaschi, Massimo},
      title = {Unleashing the Tiger: Inference Attacks on Split Learning},
      year = {2021},
      isbn = {9781450384544},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3460120.3485259},
      doi = {10.1145/3460120.3485259},
      booktitle = {Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security},
      pages = {2113–2129},
      numpages = {17},
      keywords = {ML security, deep learning, collaborative learning},
      location = {Virtual Event, Republic of Korea},
      series = {CCS '21}
}
```
