@echo off
REM Open a persistent CMD in the repo with econ CLI available
cmd /k "cd /d C:\Econometrics && doskey project=python -m econtools project $* && doskey showproject=python -m econtools project && doskey des=python -m econtools des $* && doskey summ=python -m econtools summ $* && doskey reg=python -m econtools reg $* && doskey curate=python -m econtools curate $* && doskey findcols=python -m econtools findcols $* && echo Econtools CMD ready. Use: project supervisions"
