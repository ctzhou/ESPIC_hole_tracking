# Makefile for infMagSim on a generic linux.
# Because the C-functions are in a separate file they must be compiled
# first before the python setup is called. The setup.py file includes them
# through an extra_objects parameter in the Extension. The Extension 
# process does not recognize that the extra_object is newer, so we fool
# it by touching the pyx file to ensure the cythonize is done.
# The setup.py Extension approach appears necessary to specify the 
# libraries gsl and gslcblas.

all : infMagSim_cython.so

infMagSim_cython.so: Makefile infMagSim_c.o infMagSim_cython.pyx
	touch infMagSim_cython.pyx
	python setup.py build_ext --inplace

infMagSim_c.o: infMagSim_c.c Makefile
	gcc -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
		-o infMagSim_c.o -c infMagSim_c.c

infMagSim_cython.pyx: infMagSim_cython.py
	grep -v 'utf-8' infMagSim_cython.py | grep -v nbformat | \
	 grep -v codecell | grep -v %load_ext | grep -v %cython \
                > infMagSim_cython.pyx

clean :
	rm -f *.so *.o
