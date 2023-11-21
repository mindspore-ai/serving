#/bin/bash


for i in {1..100}
do
	python client/client.py
	mv output/new.log output/6_$i.log
done

