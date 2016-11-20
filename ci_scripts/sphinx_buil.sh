#!/bin/bash
# This script is meant to be called in the "deploy" step defined in 
# circle.yml. See https://circleci.com/docs/ for more details.
# The behavior of the script is controlled by environment variable defined
# in the circle.yml in the top level folder of the project.


echo $DOC_REPO
echo $DOC_FOLDER
echo $CIRCLE_BRANCH
echo $CIRCLE_SHA1
echo $HOME
echo $USERNAME


# We just have to run the sphinx build command here 
# and commit and push
cd ..
echo "before RoBo folder"
ls
cd RoBo
echo ".... HOME ...."
ls
sphinx-build -b html ./docs/ ./docs/api
ls
echo "cd docs ........."
cd docs
ls
echo "cd api .........."
cd api
ls
cd ..
cd ..
git add docs/api
git commit -am "changes to api web files"
git push

git checkout gh-pages
echo "changed to branch gh-pages"
ls
echo "importing docs folder ....."
git checkout refactor -- docs
echo "importing complete ... "
ls
git add docs/api/
git status

cd docs
cd api
mv _sources sources
mv _static static
git add sources
git add static
cd ..
cd ..
git status
git commit -am "update api"
git push
