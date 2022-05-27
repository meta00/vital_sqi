Contributions
=============

Creating the virtual environment
--------------------------------

A virtual environment is a tool that helps to keep dependencies required by
different projects separate by creating isolated python virtual environments
for them. This is one of the most important tools that Python developers use.

In recent versions of python (>3) we can use venv. If you have various
versions of python you might need to use python3 or py instead.

.. code::

  python -m pip install venv           # Install venv
  python -m venv <environment-name>    # Create environment

Otherwise, using standard virtualenv (linux-based systems)

.. code::

  which python                                    # where is python
  pip install virtualenv                          # Install virtualenv
  virtualenv -p <python-path> <environment-name>  # create virtualenv

Let's activate the environment

.. code::

  source <environment-name>/bin/activate          # activate environment

To deactivate the environment just type

.. code::

  deactivate                                      # deactivate environment


.. warning:: Ths is slightly different on Windows systems. It is also possible
   to configure the virtual environment using the python IDE PyCharm. Students
   can get a free licence.



Fork the source repository
--------------------------

Repository: https://github.com/meta00/vital_sqi

Open the previous url and click on the fork button (top right). This creates a new
copy of the repository under your GitHub user account. The copy includes all the code,
branches, and commits from the original repo.



Set up your fork locally
------------------------

On your computer, open the terminal and create your repository folder:

.. code-block:: console

    $ mkdir vital_sqi

Move inside the folder:

.. code-block:: console

    $ cd vital_sqi

Now, lets clone the repository on your computer. You can clone the whole
repository using the following command:

.. code-block:: console

    $ git clone https://github.com/yourusername/vital_sqi.git

Or you can clone specific branches as shown below:

.. code-block:: console

    $ git clone -b main https://github.com/yourusername/vital_sqi.git
    $ mv <repository_name> main
    $ git clone -b gh-pages https://github.com/yourusername/vital_sqi.gi
    $ mv <repository_name> gh-pages

Your repository is now ready!

The main branch contains all the source files and the gh-pages will be just used
to host the documentation in html. Brief summary of the contents below:

.. code-block:: console

    gh-pages
        |- docs
            - documentation
    main
        |- docs
            |- build
            |- source
                |- conf.py    # config - sphinx documentation
                |- index.rst  # index - sphinx documentation
            make.bat
            Makefile          # run to create documentation
        |- examples
        |- pkgname            # your library
            |- core           # contains your pkg core classes
            |- tests          # contains your pkg tests - pytest
            |- utils          # contains your pkg utils


Installing your pkg in editable mode
------------------------------------

If you are planning to do any contribution, it is recommended to install the
package in editable (develop) mode. It puts a link (actually \*.pth files) into
the python installation to your code, so that your package is installed, but
any changes will immediately take effect. This way all your can import your
package the usual way.

Let's install the requirements. Move to the folder where requirements.txt is
and install all the required libraries as shown in the statements below. In
the scenario of missing libraries, just install them using pip.

.. note:: https://snarky.ca/why-you-should-use-python-m-pip/

.. code::

   python -m pip install -r requirements.txt   # Install al the requirements

Move to the directory where the setup.py is. Please note that although ``setup.py`` is
a python script, it is not recommended to install it executing that file with python
directly. Instead lets use the package manager pip.

.. warning:: Feel free to change your package name if you want. However, note that
   to make things work you will need to make the appropriate changes in existing
   files: ``setup.cfg`` and ``plot_greetings_01.py``.

.. code::

  python -m pip install --editable  .         # Install in editable mode

Read more about `packages <https://python-packaging-tutorial.readthedocs.io/en/latest/setup_py.html>`_


Generating documentation
------------------------

.. note:: To generate autodocs automatically look at sphinx-napoleon and sphinx-autodocs.
   In general the numpy documentation style is used thorough the code.

Let's use Sphinx to generate the documentation. First, you will need to install sphinx,
sphinx-gallery, sphinx-std-theme and matplotlib. Note that they might have been already
installed through the ``requirements.txt``.

Let's install the required libraries.

.. code-block:: console

  python -m pip install sphinx            # Install sphinx
  python -m pip install sphinx-gallery    # Install sphinx-gallery for examples
  python -m pip install sphinx-std-theme  # Install sphinx-std-theme CSS
  python -m pip install matplotlib        # Install matplotlib for plot examples

Then go to the docs folder within main and run:

.. code-block:: console

  make html

This will generate the documentation in the docs/build/html folder. In addition,
when using github pages it is useful to have an instruction to create the html
with sphinx and then another instruction to copy the generated htmls into the
gh-pages branch to be available online. As a shortcut, you can use

.. code-block:: console

  make github

Note that make github is defined within the Makefile and it is equivalent to:

.. code-block:: console

  make clean html
  cp -a _build/html/. ../../gh-pages/docs

These commands first generate the sphinx documentation in html and then copies
the html folder into the gh-pages branch. You can see how the documentation
looks like locally by opening the gh-pages/docs/index.html file. If you move to
the gh-pages branch and push all the changes the documentation will be also
available online thanks to GitHub Pages. You can access it through your
repository page (see Environments / GitHub Pages / Active)

Note that in order to edit the documentation you need to create .rst files and
include these newly created files in the index.rst document. An example is shown
in docs/source/tutorials/setup.rst.

In addition, you can create and document python scripts that will be automatically
included in the documentation (gallery examples) using sphinx-gallery. Remember
to include the folder(s) containing the scripts in the variable ``sphinx_gallery_conf``
in the conf.py file as shown below for tutorial.

.. code-block:: console
    :emphasize-lines: 4, 6

    # Configuration for sphinx_gallery
    sphinx_gallery_conf = {
        # path to your example scripts
        'examples_dirs': ['../../examples/tutorial'],
        # path to where to save gallery generated output
        'gallery_dirs': ['../source/_examples/tutorial'],
        # Other
        'line_numbers': True,
        'download_all_examples': False,
        'within_subsection_order': FileNameSortKey
    }

Also remember to include the .rst file automatically generated
the ``docs/index.rst`` file.

.. code-block:: console
    :emphasize-lines: 6

    .. toctree::
        :maxdepth: 2
        :caption: Example Galleries
        :hidden:

        _examples/tutorial/index

To include the output of the script (e.g. graph or console output) in the documentation
remember to prefix the script file name with ``plot`` (e.g. plot_sample_01.py). You can
find the following examples in examples/tutorial:

    - ``plot_greetings_01.py`` script using your pkgname package.
    - ``plot_sample_01.py`` script just including all the code.
    - ``plot_sample_02.py`` script documenting steps within the code.

| Read more about `sphinx <https://www.sphinx-doc.org/en/master/>`_
| Read more about `sphinx-gallery <https://sphinx-gallery.github.io/stable/index.html>`_



Running tests
-------------

Just go to the main folder and run:

.. code::

    $ pytest

You might need to install it first

.. code::

    $ python -m pip  install pytest

Read more about `pytest <https://docs.pytest.org/en/stable/>`_



Now it is time to start coding!
-------------------------------

In order to create a new contribution, please create a new branch
where name briefly explains the new feature or issue you are
addressing.


.. code::

  $ git checkout -b new_branch

And create a new remote for the upstream repo with the command:

.. code::

  $ git remote add upstream https://github.com/meta00/vital_sqi.git

Pull any new changes to keep your fork up to date:

.. code::

  $ xxx

Do any changes in your branch and create a pull request (link).

.. note:: Complete this section!

Happy coding!