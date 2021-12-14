..

====================    
LEIAP survey package
====================

This package contains various functions to help process and visualize survey information from the `Landscape, Encounters and Identity archaeology project.
<https://anthropology.washington.edu/news/2020/05/12/landscape-encounters-and-identity-project-leiap-landscape-archaeology-western>`_
  

Installation
------------

Installing miniconda
^^^^^^^^^^^^^^^^^^^^

Download and install `miniconda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html?highlight=conda>`_
for Python 3.7 or above.

- Create a project folder

- Download the ``leiap21.yml`` file into this folder by clicking on the file in the git repo. Click on **raw** button 
  and right-click to `save as...` the file onto your project folder (make sure that you save it with only the '.yml'
  extension)

- Generate a new environment using the ``leiap21.yml`` file

  >>> conda env create -f leiap21.yml

- Activate the new environment,

  >>> activate leiap21

Install **leiap_survey** package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Navigate and download the **leiap_survey** package source file into your project folder

- Use ``pip`` to install the **leiap_survey** package into the ``site-packages`` folder 
  in the *leiap_survey* environment you newly created.

  Install directly from the *source*   

  >>> pip install dist/leiap_survey-2021.9.tar.gz

  Install using *wheels*

  >>> pip install dist/leiap_survey-2021.9-py3*

Documentation
^^^^^^^^^^^^^^

Visit the following `documentation <https://mllobera.github.io/leiap_survey/docs/html/index.html>`_ page for full description of tools.  

Citation
--------
The following is part of a set of manuscripts:   

> Llobera, M, Hernández Gasch, J., Puig Palerm, A., Deppen, J., Rullán Cruellas, P., Hunt, D. and A. Iacobucci. *Interpreting Traces: A data science approach to survey data*. Submitted to  *Journal of Field Archaeology*
> Llobera, M and J. Hernández Gasch. * Making sense of difference: A new dissimilarity coefficient for clustering pottery surface assemblages*. Submitted to *Journal of Archaeological Science*  