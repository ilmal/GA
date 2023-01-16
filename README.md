# GA
code repo for "Gymnasie Arbete"


## ROADMAP

- Use MNIST dataset for metrics
- Create TfJobs (https://www.kubeflow.org/docs/components/training/tftraining/) with MNIST and deploy on k8s
- Create and deploy MNIST model with single node computation
- Deploy MNIST model with multi worker computation (https://www.tensorflow.org/guide/distributed_training#multiworkermirroredstrategy) Use mirror strat
- Run metrics on both systems (TfJobs on k8s and multi worker computation)
- Analyze metrics
- Done

## notes

- the model_save path is only passed into the app as a env variable, and so the saving needs to be handled internally in the app

- Mount a pvc and let multiple nodes write with: https://stackoverflow.com/questions/67345577/can-we-connect-multiple-pods-to-the-same-pvc

- https://github.com/kubeflow/training-operator/issues/67

- https://dzlab.github.io/ml/2020/07/18/kubeflow-training/

- https://github.com/kubeflow/training-operator/blob/master/examples/tensorflow/dist-mnist/tf_job_mnist.yaml

## SETUP PROCESS:

- I created the AI model
- I downloaded kubeflow manifests to the cluster using k8s kustomize (make sure k8s cluster is v1.21.1, kustomize is v3.2.0)
- "k create -f mnist.yaml"


### TILL NÄSTA TISDAG

- vad som behövs kodas, med containers, hur mkt som kan hämtas från kubeflow
- en mer konkret tidsplan
- se till att alla delar finns där

## REDOVISNING:

### SOURCES:

- https://towardsdatascience.com/deploying-kubeflow-to-a-bare-metal-gpu-cluster-from-scratch-6865ebcde032
- https://github.com/kubeflow/manifests/

- förklara hypotesen
- använd diagram, loss divergens i resultat


https://github.com/nottombrown/distributed-tensorflow-example/blob/master/example.py


8ps 12 work:
71 sek

2ps 4work:
90 sek


time to beat:
27.4 sek
