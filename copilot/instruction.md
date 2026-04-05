This is an simple attempt at implementing a simple STDP learning experiment. The workload is a simple unsupervised learning task in order to recognize MNIST. However, current result is abysmal. Your task is to try to identify what's wrong, produce a report, and attempt a fix.

The STDP implemented in this repo is a specific flavor of it. STDP has many flavors, with varying complexity (e.g. two-spike, one-spike, or even three-spike). But for the simple task of MNIST recognization, it shouldn't matter too much on the resulting accuracy.

Your tasks are as follows:
1. Read through the code, research relevant literature on the implementation of STDP applied to unsupervised learning and MNIST. Identify the current (attempted) implementation methods and compare them to the literature.
2. Identify the reason for the low accuracy. This could be due to a variety of reasons, such as incorrectness of the STDP, incorrectness of the unsupervised learning framework, insufficient number of neurons, empty voting class, etc. Produce a report at copilot/report.md for your findings and conclusions.
3. Produce a fix for the issue.

You are inside a NixOS container, with root privileges. You're free to install any packages with nix-env, and run any commands you like.

Auxiliary files related to this session are located in the "copilot" directory. For temporary files and scripts, please place them in `copilot/tmp`.

As always, you're encourged to ask user questions if you're unclear about anything.