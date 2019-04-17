#### tests/README.md

Code in these subdirectories runs CPU reference implementations and Gunrock GPU implementations.

To compile:
```
cd <app>
make
```

##### Getting started

The `Template` and `hello` subdirectories have annotated code that is useful for getting started.

- `Template` is an annotated/commented version of the `SSSP` app.
- `hello` is a functional app that computes the degree of each node, which is about the simplest app possible given the current API.