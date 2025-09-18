## About

GPmix is a clustering algorithm for functional data generated from Gaussian process mixtures. While designed specifically for Gaussian process mixtures, GPmix has been shown to perform well on functional data beyond this setting.

The main steps of the algorithm are:

1. **Smoothing:** Apply smoothing techniques to raw data to obtain continuous functions.  
2. **Projection:** Project the functional data onto a set of randomly generated functions.  
3. **Learning GMMs:** Fit univariate Gaussian mixture models to the projection coefficients for each projection function.  
4. **Ensemble:** Combine the multiple GMMs to extract a consensus clustering.

## Links

- [Documentation](https://gpmix.readthedocs.io/en/latest/)  
- [Source code](https://github.com/EAkeweje/GPmix)  
- [Bug reports](https://github.com/EAkeweje/GPmix/issues)

## Contributing

This project is under active development. If you encounter any bugs or have suggestions for improvements, please let us know.

Pull requests are welcome. For major changes, please open an issue first to discuss your proposed modifications. Remember to update tests as appropriate.