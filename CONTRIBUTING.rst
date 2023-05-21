============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given. Don't forget to
read and adhere to the :ref:`code-of-conduct`.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/inpefess/gym-saturation/issues

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in
  troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug"
and "help wanted" is open to whoever wants to implement a fix for it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with
"enhancement" and "help wanted" is open to whoever wants to implement
it. Contributors to the source code hold copyright of their work and
should agree to distribute it under `Apache 2.0
<https://www.apache.org/licenses/LICENSE-2.0>`__ licence.

Write Documentation
~~~~~~~~~~~~~~~~~~~

``gym-saturation`` could always use more documentation, whether as
part of the official docs, in docstrings, or even on the web in blog
posts, articles, and such. Documentation authors hold copyright of
their work and should agree to distribute it under `Apache 2.0
<https://www.apache.org/licenses/LICENSE-2.0>`__ licence.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at
https://github.com/inpefess/gym-saturation/issues.

If you are proposing a new feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to
  implement.
* Remember that this is a volunteer-driven project, and that
  contributions are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `gym-saturation` for local
development. Please note this documentation assumes you already have
`Git
<https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`__
installed and ready to go.

#. `Fork <https://github.com/inpefess/gym-saturation/fork>`__ the
   `gym-saturation` repo on GitHub.

#. Clone your fork locally:

   .. code:: sh

      cd path_for_the_repo
      git clone git@github.com:YOUR_NAME/gym-saturation.git

#. It's highly recommended to use a virtual environment for your
   local development (by the standand means of Python or using
   Anaconda or anything else):

   .. code:: bash

      python -m venv gym-saturation-env
      source gym-saturation-env/bin/activate

   This should change the shell to look something like:

   .. code:: bash

      (gym-saturation-env) $

#. Now you can install all the things you need for development:

   .. code:: bash
		   
      pip install -U pip
      pip install -U setuptools wheel poetry
      poetry install
      # install Vampire binary
      wget https://github.com/vprover/vampire/releases/download/v4.7/vampire4.7.zip -O vampire.zip
      unzip vampire.zip
      # then use vampire_z3_rel_static_HEAD_6295 as an argument or add it to $PATH
      # install iProver binary
      wget https://gitlab.com/api/v4/projects/39846772/jobs/artifacts/2023.04.10/download?job=build-job -O iprover.zip
      unzip iprover.zip
      # then use iproveropt
      # recommended but not necessary
      pre-commit install

#. Create a branch for local development:

   .. code:: bash

      git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

#. When you're done making changes, check that your changes pass code
   quality checks.

   .. code:: bash

      pydocstyle gym_saturation examples
      flake8 gym_saturation examples
      pylint gym_saturation examples
      mypy gym_saturation examples

#. The next step would be to run the test cases. `gym-saturation`
   uses pytest and all the existing tests are `doctests
   <https://docs.python.org/3/library/doctest.html>`__.

   .. code:: bash

      pytest

#. If your contribution is a bug fix or new feature, you may want to
   add a test to the existing test suite. If possible, do it by
   doctest, not a dedicates test case file.

#. Commit your changes and push your branch to GitHub:

   .. code:: bash

      git add .
      git commit -m "Your detailed description of your changes."
      git push origin name-of-your-bugfix-or-feature

#. Submit a pull request through the GitHub website.


Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these
guidelines:

#. The pull request should include tests.

#. If the pull request adds functionality, the docs should be
   updated. Put your new functionality into a function with a
   docstring, and add the feature to the list in README.rst.

#. The pull request should work for Python 3.8, 3.9, 3.10 and
   3.11. Check https://github.com/inpefess/gym-saturation/pulls and
   make sure that the tests pass for all supported Python versions.
