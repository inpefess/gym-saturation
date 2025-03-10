
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/plot_age_agent.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_plot_age_agent.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_age_agent.py:


Random and age agents for Vampire and iProver
==============================================

.. GENERATED FROM PYTHON SOURCE LINES 7-11

Random agent for Vampire
-------------------------

To make a ``gym-saturation`` environment, we have to import the package

.. GENERATED FROM PYTHON SOURCE LINES 11-16

.. code-block:: Python


    import gymnasium as gym

    import gym_saturation








.. GENERATED FROM PYTHON SOURCE LINES 17-18

then we can make a prover environment as any other Gymnasium one

.. GENERATED FROM PYTHON SOURCE LINES 18-21

.. code-block:: Python


    env = gym.make("Vampire-v0")








.. GENERATED FROM PYTHON SOURCE LINES 22-23

before using the environment, we should reset it

.. GENERATED FROM PYTHON SOURCE LINES 23-26

.. code-block:: Python


    observation, info = env.reset()








.. GENERATED FROM PYTHON SOURCE LINES 27-28

``gym-saturation`` environments don't return any ``info``

.. GENERATED FROM PYTHON SOURCE LINES 28-31

.. code-block:: Python


    print(info)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    {}




.. GENERATED FROM PYTHON SOURCE LINES 32-33

observation is a tuple of JSON representations of logic clauses

.. GENERATED FROM PYTHON SOURCE LINES 33-38

.. code-block:: Python


    from pprint import pprint

    pprint(observation)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    ({'birth_step': 0,
      'inference_parents': (),
      'inference_rule': 'input',
      'label': '1',
      'literals': 'mult(X0,mult(X1,X2)) = mult(mult(X0,X1),X2)',
      'role': 'lemma'},
     {'birth_step': 0,
      'inference_parents': (),
      'inference_rule': 'input',
      'label': '2',
      'literals': 'mult(e,X0) = X0',
      'role': 'lemma'},
     {'birth_step': 0,
      'inference_parents': (),
      'inference_rule': 'input',
      'label': '3',
      'literals': 'e = mult(inv(X0),X0)',
      'role': 'lemma'},
     {'birth_step': 0,
      'inference_parents': (),
      'inference_rule': 'input',
      'label': '4',
      'literals': 'a = mult(a,a)',
      'role': 'lemma'},
     {'birth_step': 0,
      'inference_parents': (),
      'inference_rule': 'input',
      'label': '5',
      'literals': 'e != a',
      'role': 'lemma'})




.. GENERATED FROM PYTHON SOURCE LINES 39-42

We can render the environment state in the TPTP format.
By default, we are trying to prove a basic group theory lemma:
every idempotent element equals the identity

.. GENERATED FROM PYTHON SOURCE LINES 42-45

.. code-block:: Python


    env.render()





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    cnf(1, lemma, mult(X0,mult(X1,X2)) = mult(mult(X0,X1),X2), inference(input, [], [])).
    cnf(2, lemma, mult(e,X0) = X0, inference(input, [], [])).
    cnf(3, lemma, e = mult(inv(X0),X0), inference(input, [], [])).
    cnf(4, lemma, a = mult(a,a), inference(input, [], [])).
    cnf(5, lemma, e != a, inference(input, [], [])).




.. GENERATED FROM PYTHON SOURCE LINES 46-48

here is an example of an episode during which we play random actions.
We set the random seed for reproducibility.

.. GENERATED FROM PYTHON SOURCE LINES 48-56

.. code-block:: Python


    env.action_space.seed(0)
    terminated, truncated = False, False
    while not (terminated or truncated):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
    env.close()








.. GENERATED FROM PYTHON SOURCE LINES 57-58

the episode terminated with positive reward

.. GENERATED FROM PYTHON SOURCE LINES 58-61

.. code-block:: Python


    print(terminated, truncated, reward)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    True False 1.0




.. GENERATED FROM PYTHON SOURCE LINES 62-65

It means we arrived at a contradiction (``$false``) which proves the lemma.
Notice the ``birth_step`` number of a contradiction, it shows how many steps
we did to find proof.

.. GENERATED FROM PYTHON SOURCE LINES 65-67

.. code-block:: Python

    pprint(observation[-1])





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    {'birth_step': 1077,
     'inference_parents': ('17', '5'),
     'inference_rule': 'subsumption_resolution',
     'label': '18',
     'literals': '$false',
     'role': 'lemma'}




.. GENERATED FROM PYTHON SOURCE LINES 68-72

Age agent for iProver
----------------------

We initialise iProver-based environment in the same way

.. GENERATED FROM PYTHON SOURCE LINES 72-76

.. code-block:: Python


    env = gym.make("iProver-v0")









.. GENERATED FROM PYTHON SOURCE LINES 77-78

Special magic needed if running by Jupyter

.. GENERATED FROM PYTHON SOURCE LINES 78-83

.. code-block:: Python


    import nest_asyncio

    nest_asyncio.apply()








.. GENERATED FROM PYTHON SOURCE LINES 84-86

Instead of a random agent, let's use Age agent which selects actions in the
order they appear

.. GENERATED FROM PYTHON SOURCE LINES 86-95

.. code-block:: Python


    observation, info = env.reset()
    terminated, truncated = False, False
    action = 0
    while not (terminated or truncated):
        observation, reward, terminated, truncated, info = env.step(action)
        action += 1
    env.close()








.. GENERATED FROM PYTHON SOURCE LINES 96-97

We still arrive at contradiction but it takes a different number of steps

.. GENERATED FROM PYTHON SOURCE LINES 97-100

.. code-block:: Python


    print(terminated, truncated, reward)
    pprint(observation[-1])




.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    True False 1.0
    {'birth_step': 1,
     'inference_parents': ('c_85', 'c_53'),
     'inference_rule': 'forward_subsumption_resolution',
     'label': 'c_86',
     'literals': '$false',
     'role': 'lemma'}





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 0.809 seconds)


.. _sphx_glr_download_auto_examples_plot_age_agent.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_age_agent.ipynb <plot_age_agent.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_age_agent.py <plot_age_agent.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_age_agent.zip <plot_age_agent.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
