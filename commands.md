# Commands

## Blue Crystal commands

ssh bcp3 - enter Blue Crystal

scp -r stencil bcp3 - transfer Stencil directory into Blue Crystal

module list - currently loaded modules

module av - available modules to load

module load 'moduleName' - load module

module rm 'moduleName' - unload module


## Stencil commands

make - build the code

qsub stencil.job - queue to Blue Crystal

./stencil nx ny niters - execute stencil job
./stencil 1024 1024 100
./stencil 4096 4096 100
./stencil 8000 8000 100

python check.py --ref-stencil-file stencil_1024_1024_100.pgm --stencil-file stencil.pgm - check if output correct

| nx   | ny   | niters | Reference file              |
| ---- | ---- | ------ | --------------------------- |
| 1024 | 1024 | 100    | `stencil_1024_1024_100.pgm` |
| 4096 | 4096 | 100    | `stencil_4096_4096_100.pgm` |
| 8000 | 8000 | 100    | `stencil_8000_8000_100.pgm` |

## git (local)

git add --all - if any new files are created, add them to commit

git commit -m "Name of commit" - commit files to git

git push -u origin master - upload changes to git

## git (BlueCrystal)

git pull origin master

## gprof (Profiling)

-pg - in Makefile to compile analysis file

gprof stencil - open profile data
