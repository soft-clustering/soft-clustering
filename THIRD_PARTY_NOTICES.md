# Third-party notices

SCPP is distributed under the MIT license (see `LICENSE`). Parts of it are
adapted from other open-source projects, whose notices are reproduced here
and in the source files concerned.

## NOCD reference implementation

`soft_clustering/_nocd.py` adapts the edge sampler, the graph convolutional
network, the Bernoulli-Poisson decoder, the model saver and the
early-stopping monitor of <https://github.com/shchur/overlapping-community-detection>.

```text
MIT License

Copyright (c) 2019 Oleksandr Shchur

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Other estimators

The original contributors implemented the other estimators from the cited
publications. For the 0.2.0 release:

- the git history of every module was searched for attribution (URLs,
  "adapted from", copyright or license lines); the only one found is the NOCD
  notice above;
- distinctive identifiers of the contributor-written modules that the audit
  did not rewrite (for example `COIL20Encoder` in RDFKC, `_fuzzy_and` and
  `vigilance_param` in KMART, `_update_M` and `_update_W` in RPFKM) were
  searched for on the web; the only matches were SCPP's own documentation;
- many estimators were reimplemented in the audit (see `CHANGELOG.md`), so
  their current code was written for SCPP. DMoN no longer depends on
  PyTorch Geometric: its graph convolution is SCPP's own.

These checks can show that an attribution is missing; they cannot prove
that no code was adapted. Contributors are asked to declare any third-party
origin in their pull requests (see `CONTRIBUTING.md`).
