
Summary
+++++++

cd -
sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose python-mpi4py
sudo apt-get install libgsl0ldbl gsl-bin libgsl0-dev gsl-doc-info gsl-doc-pdf gsl-ref-html gsl-ref-psdoc
sudo apt-get install cython

# I'm not sure how essential this is:
git clone git://github.com/twiecki/CythonGSL.git 
cd CythonGSL
python setup.py build
sudo python setup.py install

cd -
# If you haven't already,
git clone https://github.com/ctzhou/ESPIC_hole_tracking ESPIC
cd ESPIC
make -f Makefile
python infMagSim_script.py


