#!/bin/sh -f

docker_id=`sudo docker run \
		--privileged \
		-v $(pwd):/notebooks \
		--rm \
		-p 8888:8888 \
		--name hw07 \
		-d hw07_image` 
echo $docker_id
# Wait for 5 seconds for contaier to start 
sleep 5
sudo docker logs $docker_id
