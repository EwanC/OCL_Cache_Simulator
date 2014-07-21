#! /bin/bash

if [ -x "/usr/bin/vim" ]; then
   vim /tmp/output.cl -c " %s/int atomic_inc(__global int\*);// | wq "

else 
  echo "Please install vim"
fi
