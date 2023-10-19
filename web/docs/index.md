# Welcome to QSym²

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

??? info "A short aside"
    One must be careful when one uses Rust.
