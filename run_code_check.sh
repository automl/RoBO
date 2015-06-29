#?/bin/sh
#running all the unit tests:
python -m unittest discover robo/test -v

#static code checking:
#run after the unit tests so the code can be seen on the console
pylint robo --extension-pkg-whitelist=numpy


# make documentation 
cd docs
make html
