---
title: 'gym-saturation: an OpenAI Gym environment for saturation provers'
tags:
  - Python
  - OpenAI Gym
  - automated theorem prover
  - saturation prover
  - reinforcement learning
authors:
  - name: Boris Shminke
    orcid: 0000-0002-1291-9896
    affiliation: 1
affiliations:
 - name: Laboratoire J.A. Dieudonné, CNRS and Université Côte d'Azur, France
   index: 1
date: 1 October 2021
bibliography: paper.bib
# Summary
---

`gym-saturation` is an OpenAI Gym [@DBLP:journals/corr/BrockmanCPSSTZ16] environment for reinforcement learning (RL) agents capable of proving theorems. Currently, only theorems written in a formal language of the Thousands of Problems for Theorem Provers (TPTP) library [@Sut17] in clausal normal form (CNF) are supported. `gym-saturation` implements the 'given clause' algorithm (similar to the one used in Vampire [@DBLP:conf/cav/KovacsV13] and E Prover [@DBLP:conf/cade/0001CV19]). Being written in Python, `gym-saturation` was inspired by PyRes [@DBLP:conf/cade/0001P20]. In contrast to the monolithic architecture of a typical Automated Theorem Prover (ATP), `gym-saturation` gives different agents opportunities to select clauses themselves and train from their experience. Combined with a particular agent, `gym-saturation` can work as an ATP. Even with a non trained agent based on the clause size heuristic, `gym-saturation` can find refutations for 2245 (of 8257) CNF problems from TPTP v7.5.0.

# Statement of need

Current applications of RL to saturation-based ATPs like Enigma [@DBLP:conf/cade/JakubuvCOP0U20] or Deepire [@DBLP:conf/cade/000121a] are similar in that the environment and the agent are not separate pieces of software but parts of larger systems that are hard to disentangle. The same is true for non saturation-based RL-friendly provers too (e.g. lazyCoP, @DBLP:conf/tableaux/RawsonR21). This monolithic approach hinders free experimentation with novel machine learning (ML) models and RL algorithms and creates unnecessary complications for ML and RL experts willing to contribute to the field. In contrast, for interactive theorem provers, some projects like HOList [@DBLP:conf/icml/BansalLRSW19] separate the concepts of environment and agent. Such modular architecture proved to help create multiple successful agents by different groups of researchers (see, e.g. @DBLP:conf/aaai/PaliwalLRBS20 or @DBLP:journals/corr/abs-1905-10501). `gym-saturation` is an attempt to implement a modular environment-agent architecture of an RL-based ATP. In addition, some RL empowered saturation ATPs are not accompanied with their source code [@DBLP:journals/corr/abs-2106-03906], while `gym-saturation` is open-source software.

# Usage example

Suppose we want to prove an extremely simple theorem with a very basic agent. We can do that in the following way:

```python
# first we create and reset a OpenAI Gym environment
from importlib.resources import files
import gym

env = gym.make(
    "gym_saturation:saturation-v0",
    # we will try to find a proof shorter than 10 steps
    step_limit=10,
    # for a classical syllogism about Socrates
    problem_list=[
        files("gym_saturation").joinpath(
            "resources/TPTP-mock/Problems/TST/TST003-1.p"
        )
    ],
)
env.reset()
# we can render the environment (that will become the beginning of the proof)
print("starting hypotheses:")
print(env.render("human"))
# our 'age' agent will always select clauses for inference
# in the order they appeared in current proof attempt
action = 0
done = False
while not done:
    observation, reward, done, info = env.step(action)
    action += 1
# SaturationEnv has an additional method
# for extracting only clauses which became parts of the proof
# (some steps were unnecessary to find the proof)
print("refutation proof:")
print(env.tstp_proof)
print(f"number of attempted steps: {action}")
```

The output of this script includes a refutation proof found:

```
starting hypotheses:
cnf(p_imp_q, hypothesis, ~man(X0) | mortal(X0)).
cnf(p, hypothesis, man(socrates)).
cnf(q, hypothesis, ~mortal(socrates)).
refutation proof:
cnf(_0, hypothesis, mortal(socrates), inference(resolution, [], [p_imp_q, p])).
cnf(_2, hypothesis, $false, inference(resolution, [], [q, _0])).
number of attempted steps: 6
```

# Architecture

`gym-saturation` includes several sub-packages:

* parsing
* logic operations
* AI Gym environment implementation
* agent testing

`gym-saturation` relies on a deduction system of four rules which is known to be refutation complete [@doi:10.1137/0204036]:

\begin{align*}
{\frac{C_1\vee A_1,C_2\vee\neg A_2}{\sigma\left(C_1\vee C_2\right)}},\sigma=mgu\left(A_1,A_2\right)\quad\text{(resolution)}
\end{align*}
\begin{align*}
{\frac{C_1\vee s\approx t,C_2\vee L\left[r\right]}{\sigma\left(L\left[t\right]\vee C_1\vee C_2\right)}},\sigma=mgu\left(s,r\right)\quad\text{(paramodulation)}
\end{align*}
\begin{align*}
{\frac{C\vee A_1\vee A_2}{\sigma\left(C\vee L_1\right)}},\sigma=mgu\left(A_1,A_2\right)\quad\text{(factoring)}
\end{align*}
\begin{align*}
\frac{C\vee s\not\approx t}{\sigma\left(C\right)},\sigma=mgu\left(s,t\right)\quad\text{(reflexivity resolution)}
\end{align*}

where $C,C_1,C_2$ are clauses, $A_1,A_2$ are atomic formulae, $L$ is a literal, $r,s,t$ are terms, and $\sigma$ is a substitution (most general unifier). $L\left[t\right]$ is a result of substituting the term $r$ in $L\left[r\right]$ for the term $t$ at only one chosen position.

For parsing, we use the LARK parser [@LARK]. We represent the clauses Python classes forming tree-like structures. `gym-saturation` also includes a JSON serializer/deserializer for those trees. For example, a TPTP clause

```
cnf(a2,hypothesis,
    ( ~ q(a) | f(X) = X )).
``` 
becomes

```python
Clause(
	literals=[
		Literal(
			negated=True,
			atom=Predicate(
				name="q", arguments=[Function(name="a", arguments=[])]
			),
		),
		Literal(
			negated=False,
			atom=Predicate(
				name="=",
				arguments=[
					Function(name="f", arguments=[Variable(name="X")]),
					Variable(name="X"),
				],
			),
		),
	],
	label="a2",
)
```

This grammar serves as the glue for `gym-saturation` sub-packages, which are, in principle, independent of each other. After switching to another parser or another deduction system, the agent testing script won't break, and RL developers won't need to modify their agents for compatibility (for them, the environment will have the same standard OpenAI Gym API).

![A diagram showing interactions between four main subpackages of `gym-saturation`: 1) parsing; 2) logic operations (including the given clause algorithm); 3) OpenAI Gym Env implementation; 4) the agent testing script.\label{fig:architecture}](architecture.png)

Agent testing is a simple episode pipeline (see \autoref{fig:architecture}). It is supposed to be run in parallel (e.g. using GNU Parallel, @tange_2021_5233953) for a testing subset of problems. See the following table for the testing results of two popular heuristic-based agents on TPTP v7.5.0 with 20 steps limit:

| agent | total problems | proof found | step limit reached | error |
|--------|----------------|-------------|--------------------|-------|
| size   | 8257           | 2245        | 5889               | 123   |
| age    | 8257           | 234         | 7884               | 139   |

`size` is an agent, which always selects the shortest not yet processed clause as an action. `age` is an agent which chooses clauses in FIFO order. An error means an out-of-memory or one-hour timeout event. Trained RL agents should strive to be more successful than those primitive baselines.

# Mentions

At the moment of writing this paper, `gym-saturation` was used by its author during their PhD studies for creating experimental RL-based ATPs.

# Acknowledgements

This work has been supported by the French government, through the 3IA Côte d'Azur Investments in the Future project managed by the National Research Agency (ANR) with the reference number ANR-19-P3IA-0002.

# References
