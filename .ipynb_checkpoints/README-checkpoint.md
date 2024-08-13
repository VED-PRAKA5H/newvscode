# Github And Code Set up
1. setup the github `repository`
   * new environment  (you can do this by installing conda)
   * setup.py
   * requirements.txt

## create envienvironment 
`conda create --name IBMDB`

# 2. create setup.py file

## why to use setup.py file in machine learning project
* it is set of instructions
* it tells python things like what your project is called. who made it what it needs yo work
* without it people might have hard time to figuring out how to use
* it's like recipe for setting up your project on someone else's computer

3. # create src folder and build the package
    * in this create init.py file
    * ## why we use init.py
    * `Package Initialization`: When Python sees an init.py file inside a directory, it treats that directory as a package. It helps organize your code into logical units. 
    * `Namespace Package`:  If you have a large project split across multiple directories or locations, each having its own init.py, Python combines them all together when you import the package. This allows you to spread your code across different parts of your project without conflicts.m.

4. # create requirements.txt
    * ## -e stand for specifies that the package should be installed in editable mode.
    * ## and . for current directory
    * both are use in `pip` not in `conda` (for conda: Remove the -e . line)
```
pandas
numpy
matplotlib
seaborn
scikit-learn
-e .
``` 

5. # now run `python setup.py install` in cmd of youyour virtual environment
   ## now you can use `pip freeze > requirements.txt` for conda `conda list --export > requirements.txt`

# project stucture, logging and exceptionn handling 

