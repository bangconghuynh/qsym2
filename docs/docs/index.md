---
title: QSym²
description: "QSym²: a program for Quantum Symbolic Symmetry"
hide:
  - navigation
---

<style>
    .md-typeset h1 {
        display: none;
    }
</style>

<figure markdown>
  ![QSym²](assets/logos/qsym2_logo_no_background_light.svg#only-dark){ width="300" }
  ![QSym²](assets/logos/qsym2_logo_no_background_dark.svg#only-light){ width="300" }
</figure>

<p style="text-align: center;">
  A program for <b>Q</b>uantum <b>Sym</b>bolic <b>Sym</b>metry
</p>

-------

For full documentation visit [mkdocs.org](https://www.mkdocs.org).

## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.

## Test code highlighting

This is Rust:
```rust linenums="1" hl_lines="2"
let a: Vec<u32> = vec![0, 1, 2]; // (1)!
let b = a.iter().map(|x| x * 2).collect::<Vec<_>>(); // (2)!
```

1. This is to define a *new* variable called `a`.
2. This iterates over `a` and defines a new variable called `b`.

???+ warning "A short aside"
    One must be careful when one uses Rust.

These are different ways of achieving the same thing:
=== "Rust"
    ```rust
    let a = vec![1, 2, 3];
    ```

=== "Python"
    ```python
    a = [1, 2, 3]
    ```

## Test maths

This is an inline equation: $E = mc^2$.

This is a display equation:

$$
    \int_{-\infty}^{\infty} \exp(-r^2) dr.
$$
