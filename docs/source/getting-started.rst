Getting Started
===============

Overview
--------

GPmix is a clustering algorithm for functional data that are generated from Gaussian process mixtures. Although designed for Gaussian process mixtures, our experimental study demonstrated that GPmix works well even for functional data that are not from Gaussian process mixtures.

The main steps of the algorithm are:

1. **Smoothing:** Apply smoothing methods on the raw data to get continuous functions.
2. **Projection:** Project the functional data onto a few randomly generated functions.
3. **Learning GMMs:** For each projection function, learn a univariate Gaussian mixture model from the projection coefficients.
4. **Ensemble:** Extract a consensus clustering from the multiple GMMs.

If you use this package in your research, please cite:

.. code-block:: bibtex

   @InProceedings{pmlr-v235-akeweje24a,
     title = {Learning Mixtures of {G}aussian Processes through Random Projection},
     author = {Akeweje, Emmanuel and Zhang, Mimi},
     booktitle = {Proceedings of the 41st International Conference on Machine Learning},
     pages = {720--739},
     year = {2024},
     editor = {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
     volume = {235},
     series = {Proceedings of Machine Learning Research},
     month = {21--27 Jul},
     publisher = {PMLR},
   }

Installation
------------

**Python version support:** GPmix requires **Python 3.9 or newer**. To install GPmix:

.. code-block:: bash

   pip install GPmix

Contributing
------------

Contributions are welcome! Fork the project on `GitHub <https://github.com/EAkeweje/GPmix.git>`_ and submit a pull request.

Next Steps
----------

- Read the :doc:`quick-start` guide for a beginner's guide on how to apply GPmix clustering to your data.
- Read the :doc:`advanced-usage` guide for more advanced usage and examples.
- Check out the :ref:`API Reference <api>` for details.