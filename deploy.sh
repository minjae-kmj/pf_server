#!/bin/sh

project=dr-folio
service_name=flask-server
tag=latest
image=$service_name:$tag
region=asia-northeast3


set -x

# Deploy image
gcloud builds submit --tag gcr.io/$project/$image

# Deploy cloud run
gcloud run deploy $service_name --image gcr.io/$project/$image --region $region
``