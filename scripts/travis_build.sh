REGISTRY_URL=us.icr.io
NAMESPACE=cil15-shared-registry
ARCH=amd64

if [ $TRAVIS_BRANCH = "main" ]
then
    # login into cil15-registry
    echo $IBM_CLOUD_API_KEY | docker login -u iamapikey --password-stdin $REGISTRY_URL

    if [ $BUILD_DEEPSPEED_IMAGE = "true" ]
    then
        IMAGE=$REGISTRY_URL/$NAMESPACE/deepspeed:$DEEPSPEED_IMAGE_TAG
        docker build -f docker/deepspeed.dockerfile -t $IMAGE --platform $ARCH .
        docker push $IMAGE
    fi
fi
