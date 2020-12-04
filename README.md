# Unleashing the Tiger: Inference Attacks on Split Learning
We investigate the security of **split learning**---a novel collaborative machine learning framework that enables peak performance by requiring minimal resources consumption. In the paper, we make explicit the vulnerabilities of the protocol and demonstrate its inherent insecurity by introducing general attack strategies targeting the reconstruction of clients' private training sets. More prominently, we demonstrate that a malicious server can actively hijack the learning process of the distributed model and bring it into an insecure state that enables inference attacks on clients' data. We implement different adaptations of the attack and test them on various datasets as well as within realistic threat scenarios.

## Code:

This repository contains all the code needed to run the **Feature-space hijacking attack** and its variations. The implementation is based on **TensorFlow2**.

In particular:

*  *FSHA.py*: It implements the attack and a single-user version of split learning. The class *FSHA_binary_property* in the file implements the property-inference attack.
* *architectures.py*: It contains the main network architectures we used in the paper.
* *styleloss.py*: It implements the style loss based on a MobileNet trained on *ImageNet*.
* *datasets.py*: It contains utility to load and parse datasets.

## Proof Of Concept 🐯:

We report a set of *jupyter notebooks* that act as brief tutorial for the code and replicate the experiments in the paper. Those are:

* *FSHA.ipynb*: It implements the standard Feature-space hijacking attack on the MNIST dataset.
* *FSHA_mangled.ipynb*: It implements the Feature-space hijacking attack on the MNIST dataset, when the attacker's training set is mangled of a class.
* Work in progress.....

## Cite our work:
to be uploaded.
> @InProceedings{unleashingthetiger,<br>
> author="Pasquini, Dario and Ateniese, Giuseppe and Bernaschi, Massimo",<br>
> title="Unleashing the Tiger: Inference Attacks on Split Learning",<br>
> year="2020",<br>
> }



