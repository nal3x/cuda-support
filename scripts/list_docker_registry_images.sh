#!/bin/bash

source ./global.sh

# If you have a registry that requires authentication in order to list
# the available images, then you can use this script to list them.
# 
# How to use this script:
# run it like this: ./list_docker_registry_images.sh <username> <password>
# 
# IMPORTANT: Use only from within trusted networks, and ONLY for debugging, 
#			 since the script will NOT use an encrypted communication channel.

if [[ $# -ne 2 ]] ; then 
	echo "Wrong number of input arguments!"
	echo "Please provide username and password for the docker registry"
	echo "Example: $ ./$(basename "$0") <USERNAME> <PASSWORD>"
	exit 0
fi

DR_USERNAME=$1
DR_PASSWORD=$2

# Identify available images
catalog=$(curl -X GET -u "${DR_USERNAME}":"${DR_PASSWORD}" https://"${DOCKER_REGISTRY}"/v2/_catalog 2> /dev/null)
if echo "$catalog" | grep -q "401 Unauthorized"; then
	echo "Wrong credentials provided. Please give the correct username and password"
	exit 0
fi
images=$(echo "$catalog" | cut -d'[' -f 2 | cut -d']' -f 1 | tr -d '\"')
IFS=', ' read -r -a images_array <<< "$images"

for img in "${images_array[@]}" ; do
	# Identify versions
	img_versions=$(curl -X GET -u "${DR_USERNAME}":"${DR_PASSWORD}" https://"${DOCKER_REGISTRY}"/v2/"${img}"/tags/list 2> /dev/null)
	versions_txt=$(echo "$img_versions" | cut -d'[' -f 2 | cut -d']' -f 1 | tr -d '\"')
	IFS=', ' read -r -a versions_array <<< "$versions_txt" 
	

	# List images along with their versions
	for version in "${versions_array[@]}" ; do
		echo "${DOCKER_REGISTRY}/${img}:${version}"
	done
done

