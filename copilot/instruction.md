This is an simple attempt at implementing a simple STDP learning experiment. The workload is a simple unsupervised learning task in order to recognize MNIST. However, current result is abysmal. Your task is to try to identify what's wrong, produce a report, and attempt a fix.

The STDP implemented in this repo is a specific flavor of it. STDP has many flavors, with varying complexity (e.g. two-spike, one-spike, or even three-spike). But for the simple task of MNIST recognization, it shouldn't matter too much on the resulting accuracy.

Your task is to compare the implementation with the canonical literature on this topic.
1. The code in https://github.com/peter-u-diehl/stdp-mnist is added as a submodule in ref/. That's the code for the paper "Unsupervised Learning of Digit Recognition using Spike-Timing-Dependent Plasticity" by Diehl and Cook (2015).
2. Run the code with the downloaded data from previous sessions. See if we can reproduce the result in the paper.
3. Compare the implementation with our implementation. Identify the differences, understand the effect of these differences, identify which differences are likely to cause the difference in accuracy, and produce a report on the findings in `copilot/report.diff.md`.
4. Attempt to fix the issue in our codebase, and produce a report on the fix in `copilot/fix.diff.md`. See if we recover the accuracy after the fix.

You are inside a NixOS container, with root privileges. You're free to install any packages with nix-env, and run any commands you like.

Auxiliary files related to this session are located in the "copilot" directory. For temporary files and scripts, please place them in `copilot/tmp`.

As always, you're encourged to ask user questions if you're unclear about anything.

## Previous instructions

### 1.5. Additional architectural changes

several sessions are performed with direct prompts. The log for those sessions can be found in `copilot/report.md`, `copilot/improvement.md`, and `copilot/improvement.2.md`.

### 1. Initial diganosis

Your tasks are as follows:
1. Read through the code, research relevant literature on the implementation of STDP applied to unsupervised learning and MNIST. Identify the current (attempted) implementation methods and compare them to the literature.
2. Identify the reason for the low accuracy. This could be due to a variety of reasons, such as incorrectness of the STDP, incorrectness of the unsupervised learning framework, insufficient number of neurons, empty voting class, etc. Produce a report at copilot/report.md for your findings and conclusions.
3. Produce a fix for the issue.