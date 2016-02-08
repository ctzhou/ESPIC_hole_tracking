infMagSim: infMagSim_script.py infMagSim_cython.so

#infMagSim_script.py: infMagSim.py
#	sed -n '/#####ScriptStart#####/,/#####ScriptEnd#####/p' infMagSim.py \
		| grep -v '#####Script' | grep -v codecell | grep -v %%px \
		> infMagSim_script.py

infMagSim_cython.pyx: infMagSim_cython.py
	grep -v 'utf-8' infMagSim_cython.py | grep -v nbformat | grep -v codecell \
		| grep -v %load_ext | grep -v %cython \
		> infMagSim_cython.pyx

infMagSim_cython.c: infMagSim_cython.pyx
	cython infMagSim_cython.pyx

infMagSim_cython.o: infMagSim_cython.c
	gcc -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
		-I /home/chaako/local/include/ \
		-I /home/chaako/virtualenvs/IPython/include/python2.7/ \
		-I /home/chaako/virtualenvs/IPython/lib/python2.7/site-packages/numpy/core/include/ \
		-o infMagSim_cython.o -c infMagSim_cython.c \
		-L /home/chaako/local/lib -lgsl -lgslcblas 

infMagSim_c.o: infMagSim_c.c
	gcc -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
		-o infMagSim_c.o -c infMagSim_c.c

infMagSim_cython.so: infMagSim_cython.o infMagSim_c.o
	gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
		-o infMagSim_cython.so infMagSim_cython.o infMagSim_c.o \
		-L /home/chaako/local/lib -lgsl -lgslcblas 
clean:
	rm *.o *.so
