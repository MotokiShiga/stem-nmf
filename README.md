## Nonnegative Matrix Factorization for STEM-EELS/EDX Analysis

This repository provides MATLAB and Python codes of our proposed methods in [1].

In MATLAB, you can run a demo script for **NMF-SO** (Nonnegative Matrix Factorization with Soft Orthogonality constraint):

    demo_nmf_so

and **NMF-ARD-SO** (Nonnegative Matrix Factorization with Automatic Relevance Determination and Soft Orthogonality constraint):

    demo_nmf_ard_so
    
**SO** is for resolving spatial overlaps among chemical components and **ARD** is for optimizing the number of chemical components.

Our python library code (supported on Python 3.5.1+) was updated on July 10, 2017. The new code defines a class for each NMF model and use method **fit** to learn, similarly to scikit-learn. See jupyter notebook **demo_libnmf.ipynb**. 


## Reference

[1]
Motoki Shiga, Kazuyoshi Tatsumi, Shunsuke Muto, Koji Tsuda, Yuta Yamamoto, Toshiyuki Mori, Takayoshi Tanji,
  **"Sparse Modeling of EELS and EDX Spectral Imaging Data by Nonnegative Matrix Factorization"**,  
  *Ultramicroscopy*, Vol.170, p.43-59, 2016.  
[http://dx.doi.org/10.1016/j.ultramic.2016.08.006](http://dx.doi.org/10.1016/j.ultramic.2016.08.006)


## License

MIT License (see `LICENSE` file).
