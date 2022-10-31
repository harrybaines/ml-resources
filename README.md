# ml-resources

This repo contains all the code and resources I've collected over the years and provides a handy single point 
of reference for most ML or general data science projects. 

## Creating a Virtual Environment

First, create a new environment called `venv`:

```bash
python -m venv venv
```

Then activate it:

```bash
source venv/bin/activate
```

Then install dependencies into it (if you have a `requirements.txt`, otherwise just use `pip install X Y Z`):

```bash
(venv) pip install -r requirements.txt
```

If you want to use Jupyter, install `ipykernel`:

```bash
(venv) python -m ipykernel install --user --name=venv
```

Remember to select `venv` as the preferred kernel in Jupyter so it can read your dependencies.

## Useful Repositories

- [General data science template in Python](https://github.com/harrybaines/data-science-template)

## Maintenance

To update the submodules in this repo:

```bash
git submodule foreach git pull origin main
```

Then add and commit the new changes to create a new commit for the updated submodules in this 
repo.
