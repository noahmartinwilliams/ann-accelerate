# ann-accelerate library

## What is this?

This library provides simple artificial neural networks that can be used on GPUs via the accelerate library.

## Compilation

You'll need to compile accelerate and accelerate-llvm. You'll need to use the updated version of language-c which can be found [here](https://github.com/noahmartinwilliams/language-c).
Then you'll need to compile [c2hs](https://github.com/noahmartinwilliams/c2hs) using the language-c library with cabal.project files.

## Usage

Examples of how to use this library can be found in the tests (which take the form of the executables included in this repo).
