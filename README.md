

# cr_pool

## Installation
We recommend to use the [`uv`](https://github.com/astral-sh/uv) package manager for installing all
python dependencies.
The following instructions will work for unix-based operating systems.
We use [`maturin`](https://github.com/PyO3/maturin) as a build tool.

```
# Create a new virtual environment and activate it
python -m venv .venv
source .venv/bin/activate

# Install maturin if not already present
uv pip install maturin

# Install this package and all its dependencies
# Prepend --uv if you are using the uv package manager
# otherwise omit this flag
python -m maturin develop -r --uv
```

## TODO

- Formatting
    - [x] Use double Column layout?

-  Abstract

-  Introduction

-  General Math
    - [x] Move to supplement

-  One-Species
    - [x] Remove matrix formulation
    - [x] Page 3: Remove bullet points in front of pools; mention "the fraction of the population" in
      sentence before
    - [x] Page 4: remove matrix of system (put supplement)
    - [x] Back flow from Growth to Lag pool: make to paragraph "Maxwell type of stress ..."

-  Interaction between Species
    - [x] Change in supplement too!
    - [x] Link to supplement for Lotka-Volterra discussion
    - [x] Mutual inhibition: Rename Toxin variable T to generalized inhibition I such that it matches
      with spatial effects

-  Spatial Effects
    - [x] Page 16: R* = R / R_0 probably remove it
    - [x] Rename variables: G_max -> N_t and R* -> R
        - [x] make sure that they match to previous sections
    - [ ] More Citations
    - [x] Figure 8: Do we need this?
    - [x] Table 1: Move to supplement
    - [ ] Ensure that we are talking about pool model! (map everything on this new theoretical framework)
    - [ ] Use introductory part in main Introduction as well
    - [x] Discuss results in text form not only in figure
    - [ ] Redo plots; use pgf backend; do not use combined bacteria volume; use N/N_t or similar

-  Discussion
    - [ ] difference in lag phase between baranyi-roberts and pool model
    - [ ] mathematical generalization has only linear effects: is this justified? (do we include this?)
    - [ ] how can we describe other effects
        - [ ] more states for different metabolic processes
        - [ ] more pools for more species
        - [ ] other cooperation/inhibition effects
        - [ ] external influences (i.e. put drug into system)
    - [ ] assumptions in setting up ABM model; which details do we include?

-  Conclusion
    - [ ] new theoretical framework: pool model
    - [ ] consistent with ODE models (Baranyi Roberts)
    - [ ] pool model can explain better than known models
    - [ ] account + understand for many other effects such as inhibition and toxins
    - [ ] showed spatial limitation of the model

-  Supplement
    - [x] Formatting of Supplement subsections
    - [x] Bundle all existing subsections into "ABM Details"
    - [x] "General Mathematical Formalism"
        - [x] Move section 2 here
        - [x] Move all matrix equations here
        - [ ] Discuss generalized Lotka-Volterra model here extensively!
            - [ ] Christian should probably write this text

- Move ABM simulation Code to repository

## Order of TODO
1. Content
    - [x] One-Species
    - [x] Interaction between Species
    - [ ] Spatial Effects
    - [x] Supplement
2. Discussion
3. Formal stuff
    - [ ] Abstract
    - [ ] Introduction
    - [ ] Discussion
    - [ ] Conclusion
4. Review of whole manuscript
    - [ ] Formulations
    - [x] Notation
    - [ ] Formatting
