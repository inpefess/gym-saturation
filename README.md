# Saturation Gym

Saturation Gym is an [OpenAI Gym](https://gym.openai.com/) environment for saturation provers (like [PyRes](https://github.com/eprover/PyRes)).

# Installation

```sh
pip install gym-saturation
```

# Usage

See `examples/example.ipynb` notebook for more information.

# What is going on

One can write theorems in a machine-readable form. This package uses the CNF sublanguage of TPTP. Before using the environment, you will need to download a recent TPTP archive (ca 600MB).

A statement of a theorem becomes a list of clauses. In a given clause algorithm, one divides the clauses in processed and not processed yet. Then at each step, one selects a not processed yet clause as a given clause. If it's empty (we arrived at a contradiction, i.e. found a refutation proof), the algorithm stops with success. If not, one computes all possible resolvents (the results of applying a simple but powerful resolution deduction rule to the given clause and all processed clauses). Then we add resolvents to the unprocessed set, and the given clause goes into the processed. The algorithm iterates if we didn't run out of time and unprocessed clauses.

For the choice of a given clause, one usually employs a clever combination of heuristics. Of course, we can reformulate the same process as a reinforcement learning task.

# What is a State

(More or less resembles [`ProofState` class of PyRes](https://github.com/eprover/PyRes/blob/master/saturation.py))

The environment's state is a list of logical clauses. Each clause is a list of literals and also has several properties:

* `label` --- comes from the problem file or starts with `inferred_` if inferred during the episode
* `processed` --- boolean value splitting clauses into unprocessed and processed ones; in the beginning, everything is not processed
* `birth_step` --- a number of the step when the clause appeared in the unprocessed set; clauses from the problem have `birth_step` zero
* `inference_parents` --- a list of labels from which the clause was inferred. For clauses from the problem statement, this list is empty.

Literal is a predicate, negated or not. A predicate can have arguments, which can be functions or variables. Functions can have arguments, which in turn can be functions or variables.

Grammar is encoded in Python objects in a self-explanatory way. Each grammar object is a dictionary with an obligatory key `class` (`Clause`, `Literal`, `Predicate`, `Function`, `Variable`), and other keys representing this object's properties (such as being negated or having a list of arguments). To parse these JSON representation into package's inner representation, use `gym_saturation.parsing.json_grammar.dict_to_clause`.

# What is an Action

Action is an index of a clause from the state. Valid actions are only indices of not processed clauses.

# What is a Reward

`1.0` if the proof is found (a clause with an empty list of literals is selected as an action).

`-1.0`, if all clauses from the state are processed or step limit is reached, but no proof is found

`0.0` in any other case (valid action chosen, saturation step performed, step limit not reached yes, no proof is found, there are still not processed clauses)

# Important notice

Usually, saturation provers use a timeout in seconds since they work in real-time mode. Here, we live in a discrete time, so we limit a prover by the number of saturation algorithm steps taken, not wall-clock time.
