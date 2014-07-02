ffd
===

ffd is a MATLAB toolbox that implements the factored form descent algorithm, the original form of which was described in [1]. It can be used to retrieve the spatial coherence statistics of a quasi-monochromatic source given known optical system(s) and measured point-wise intensity values of the output of said field through said optical system(s).

Installation
---

Add the directory containing ffd.m to MATLAB's path using addpath or something similar. Make sure the entire directory structure is the same as the repository (e.g. +ffd directories are present, etc.). Run the provided sample code to test your installation:

```Matlab
>> ffd.sample.run
```

References
---

1. Z. Zhang, Z. Chen, S. Rehman, and G. Barbastathis, "Factored form descent: a practical algorithm for coherence retrieval," Opt. Express **21**, 5759--5780 (2013)
