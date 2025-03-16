.. _development-guide:

=================
Development Guide
=================

Get Started!
------------

Ready to contribute? Here's how to set up `gym-saturation` for local
development. Please note this documentation assumes you already have
`Git
<https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`__
installed and ready to go.

#. `Fork <https://github.com/inpefess/gym-saturation/fork>`__ the
   `gym-saturation` on GitHub.

#. Clone your fork locally:

   .. code:: sh

      cd git_URL
      git clone git@github.com:YOUR_NAME/gym-saturation.git

#. Install
   [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)

#. It's highly recommended to use a virtual environment for your
   local development (by the standard means of Python or using
   Anaconda or anything else):

   .. code:: bash

      python -m venv gym-saturation-env
      source gym-saturation-env/bin/activate

   This should change the shell to look something like:

   .. code:: bash

      (gym-saturation-env) $

#. Now you can install all the things you need for development:

   .. code:: bash
		   
      poetry install --all-groups
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

      git checkout -b name-of-your-bug-fix-or-feature

   Now you can make your changes locally.

#. When you're done making changes, check that your changes pass code
   quality checks.

   .. code:: bash

      pydocstyle gym_saturation
      flake8 gym_saturation
      pylint gym_saturation
      mypy gym_saturation

#. The next step would be to run the test cases. `gym-saturation`
   uses pytest and all the existing tests are `doctest
   <https://docs.python.org/3/library/doctest.html>`__.

   .. code:: bash

      coverage run -m pytest
      coverage report -m

#. If your contribution is a bug fix or new feature, you may want to
   add a test to the existing test suite. If possible, do it by
   doctest, not a dedicates test case file.

#. Commit your changes and push your branch to GitHub:

   .. code:: bash

      git add .
      git commit -m "Your detailed description of your changes."
      git push origin name-of-your-bug-fix-or-feature

#. Submit a pull request through the GitHub website.


Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these
guidelines:

#. The pull request should include tests.

#. If the pull request adds functionality, the docs should be
   updated. Put your new functionality into a function with a
   docstring, and add new classes or functions to a relevant file in
   the `doc/api` folder. To build the doc locally:

   .. code:: bash

       cd doc
       make html
   
#. The pull request should work for Python 3.9, 3.10, 3.11, 3.12 and
   3.13. Check https://github.com/inpefess/gym-saturation/pulls and
   make sure that the CI checks pass for all supported Python
   versions.
