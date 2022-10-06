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

### command dumps:

```
nils@NilsOld:/mnt/cs/GA/kubeflow/manifests$ k get all
NAME                                                         READY   STATUS             RESTARTS   AGE
pod/admission-webhook-deployment-c77d48bbb-nrhw6             1/1     Running            0          9m9s
pod/cache-server-56d94f5d78-drvsc                            2/2     Running            0          9m9s
pod/centraldashboard-5864f74d99-jsxzr                        2/2     Running            0          9m8s
pod/jupyter-web-app-deployment-5bc998bcb5-rwkgm              1/1     Running            0          9m8s
pod/katib-controller-6848d4dd9f-wjbqd                        1/1     Running            0          9m8s
pod/katib-db-manager-665954948-pmw5b                         0/1     CrashLoopBackOff   5          9m8s
pod/katib-mysql-5bf95ddfcc-2m6s5                             0/1     Pending            0          9m7s
pod/katib-ui-56ccff658f-vc7pr                                1/1     Running            0          9m7s
pod/kserve-controller-manager-0                              2/2     Running            0          8m45s
pod/kserve-models-web-app-5878544ffd-4twjr                   2/2     Running            0          9m7s
pod/kubeflow-pipelines-profile-controller-5d98fd7b4f-j8csp   1/1     Running            0          9m7s
pod/metacontroller-0                                         1/1     Running            0          8m45s
pod/metadata-envoy-deployment-5b685dfb7f-mvb2g               1/1     Running            0          9m7s
pod/metadata-grpc-deployment-f8d68f687-bwfbp                 1/2     CrashLoopBackOff   4          9m6s
pod/metadata-writer-d6498d6b4-fp9jt                          1/2     Error              2          9m6s
pod/minio-5b65df66c9-pq5cw                                   0/2     Pending            0          9m6s
pod/ml-pipeline-844c786c48-nbjtl                             1/2     Running            3          9m6s
pod/ml-pipeline-persistenceagent-5854f86f8b-v5cjl            2/2     Running            0          9m6s
pod/ml-pipeline-scheduledworkflow-5dddbf664f-2rcvw           2/2     Running            0          9m6s
pod/ml-pipeline-ui-6bdfc6dbcd-zv7qb                          2/2     Running            0          9m6s
pod/ml-pipeline-viewer-crd-85f6fd557b-msq6b                  2/2     Running            1          9m6s
pod/ml-pipeline-visualizationserver-7c4885999-58xz4          2/2     Running            0          9m5s
pod/mysql-5c7f79f986-zt8hh                                   0/2     Pending            0          9m5s
pod/notebook-controller-deployment-6478d4858c-49bl5          2/2     Running            0          9m5s
pod/profiles-deployment-7bc47446fb-tnqf2                     3/3     Running            1          9m5s
pod/tensorboard-controller-deployment-f4f555b95-g2c89        3/3     Running            1          9m4s
pod/tensorboards-web-app-deployment-7578c885f7-nlmvs         1/1     Running            0          9m4s
pod/training-operator-6c9f6fd894-2qlv4                       1/1     Running            0          9m4s
pod/volumes-web-app-deployment-7bc5754bd4-7jwlc              1/1     Running            0          9m3s
pod/workflow-controller-6b9b6c5b46-7lk7t                     2/2     Running            1          9m3s

NAME                                                                TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)             AGE
service/admission-webhook-service                                   ClusterIP   10.100.130.114   <none>        443/TCP             10m
service/cache-server                                                ClusterIP   10.104.112.145   <none>        443/TCP             10m
service/centraldashboard                                            ClusterIP   10.108.80.75     <none>        80/TCP              10m
service/jupyter-web-app-service                                     ClusterIP   10.97.57.96      <none>        80/TCP              10m
service/katib-controller                                            ClusterIP   10.106.158.222   <none>        443/TCP,8080/TCP    10m
service/katib-db-manager                                            ClusterIP   10.107.1.201     <none>        6789/TCP            10m
service/katib-mysql                                                 ClusterIP   10.101.243.4     <none>        3306/TCP            10m
service/katib-ui                                                    ClusterIP   10.101.58.37     <none>        80/TCP              10m
service/kserve-controller-manager-metrics-service                   ClusterIP   10.106.11.75     <none>        8443/TCP            10m
service/kserve-controller-manager-service                           ClusterIP   10.103.64.105    <none>        8443/TCP            10m
service/kserve-models-web-app                                       ClusterIP   10.111.127.107   <none>        80/TCP              10m
service/kserve-webhook-server-service                               ClusterIP   10.106.121.57    <none>        443/TCP             10m
service/kubeflow-pipelines-profile-controller                       ClusterIP   10.109.14.21     <none>        80/TCP              10m
service/metadata-envoy-service                                      ClusterIP   10.99.171.177    <none>        9090/TCP            10m
service/metadata-grpc-service                                       ClusterIP   10.106.113.36    <none>        8080/TCP            10m
service/minio-service                                               ClusterIP   10.96.235.117    <none>        9000/TCP            10m
service/ml-pipeline                                                 ClusterIP   10.111.28.35     <none>        8888/TCP,8887/TCP   10m
service/ml-pipeline-ui                                              ClusterIP   10.100.142.65    <none>        80/TCP              10m
service/ml-pipeline-visualizationserver                             ClusterIP   10.104.60.232    <none>        8888/TCP            10m
service/mysql                                                       ClusterIP   10.109.169.87    <none>        3306/TCP            10m
service/notebook-controller-service                                 ClusterIP   10.106.22.29     <none>        443/TCP             10m
service/profiles-kfam                                               ClusterIP   10.106.194.251   <none>        8081/TCP            10m
service/tensorboard-controller-controller-manager-metrics-service   ClusterIP   10.102.194.207   <none>        8443/TCP            10m
service/tensorboards-web-app-service                                ClusterIP   10.108.9.154     <none>        80/TCP              10m
service/training-operator                                           ClusterIP   10.100.242.24    <none>        8080/TCP            10m
service/volumes-web-app-service                                     ClusterIP   10.102.16.184    <none>        80/TCP              10m
service/workflow-controller-metrics                                 ClusterIP   10.110.201.249   <none>        9090/TCP            10m

NAME                                                    READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/admission-webhook-deployment            1/1     1            1           10m
deployment.apps/cache-server                            1/1     1            1           10m
deployment.apps/centraldashboard                        1/1     1            1           10m
deployment.apps/jupyter-web-app-deployment              1/1     1            1           10m
deployment.apps/katib-controller                        1/1     1            1           10m
deployment.apps/katib-db-manager                        0/1     1            0           10m
deployment.apps/katib-mysql                             0/1     1            0           10m
deployment.apps/katib-ui                                1/1     1            1           10m
deployment.apps/kserve-models-web-app                   1/1     1            1           10m
deployment.apps/kubeflow-pipelines-profile-controller   1/1     1            1           10m
deployment.apps/metadata-envoy-deployment               1/1     1            1           10m
deployment.apps/metadata-grpc-deployment                0/1     1            0           10m
deployment.apps/metadata-writer                         0/1     1            0           10m
deployment.apps/minio                                   0/1     1            0           10m
deployment.apps/ml-pipeline                             0/1     1            0           10m
deployment.apps/ml-pipeline-persistenceagent            1/1     1            1           10m
deployment.apps/ml-pipeline-scheduledworkflow           1/1     1            1           10m
deployment.apps/ml-pipeline-ui                          1/1     1            1           10m
deployment.apps/ml-pipeline-viewer-crd                  1/1     1            1           10m
deployment.apps/ml-pipeline-visualizationserver         1/1     1            1           10m
deployment.apps/mysql                                   0/1     1            0           10m
deployment.apps/notebook-controller-deployment          1/1     1            1           10m
deployment.apps/profiles-deployment                     1/1     1            1           10m
deployment.apps/tensorboard-controller-deployment       1/1     1            1           10m
deployment.apps/tensorboards-web-app-deployment         1/1     1            1           10m
deployment.apps/training-operator                       1/1     1            1           10m
deployment.apps/volumes-web-app-deployment              1/1     1            1           10m
deployment.apps/workflow-controller                     1/1     1            1           10m

NAME                                                               DESIRED   CURRENT   READY   AGE
replicaset.apps/admission-webhook-deployment-c77d48bbb             1         1         1       10m
replicaset.apps/cache-server-56d94f5d78                            1         1         1       10m
replicaset.apps/centraldashboard-5864f74d99                        1         1         1       10m
replicaset.apps/jupyter-web-app-deployment-5bc998bcb5              1         1         1       10m
replicaset.apps/katib-controller-6848d4dd9f                        1         1         1       10m
replicaset.apps/katib-db-manager-665954948                         1         1         0       10m
replicaset.apps/katib-mysql-5bf95ddfcc                             1         1         0       10m
replicaset.apps/katib-ui-56ccff658f                                1         1         1       10m
replicaset.apps/kserve-models-web-app-5878544ffd                   1         1         1       10m
replicaset.apps/kubeflow-pipelines-profile-controller-5d98fd7b4f   1         1         1       10m
replicaset.apps/metadata-envoy-deployment-5b685dfb7f               1         1         1       10m
replicaset.apps/metadata-grpc-deployment-f8d68f687                 1         1         0       10m
replicaset.apps/metadata-writer-d6498d6b4                          1         1         0       10m
replicaset.apps/minio-5b65df66c9                                   1         1         0       10m
replicaset.apps/ml-pipeline-844c786c48                             1         1         0       10m
replicaset.apps/ml-pipeline-persistenceagent-5854f86f8b            1         1         1       10m
replicaset.apps/ml-pipeline-scheduledworkflow-5dddbf664f           1         1         1       10m
replicaset.apps/ml-pipeline-ui-6bdfc6dbcd                          1         1         1       10m
replicaset.apps/ml-pipeline-viewer-crd-85f6fd557b                  1         1         1       10m
replicaset.apps/ml-pipeline-visualizationserver-7c4885999          1         1         1       10m
replicaset.apps/mysql-5c7f79f986                                   1         1         0       10m
replicaset.apps/notebook-controller-deployment-6478d4858c          1         1         1       10m
replicaset.apps/profiles-deployment-7bc47446fb                     1         1         1       10m
replicaset.apps/tensorboard-controller-deployment-f4f555b95        1         1         1       10m
replicaset.apps/tensorboards-web-app-deployment-7578c885f7         1         1         1       10m
replicaset.apps/training-operator-6c9f6fd894                       1         1         1       10m
replicaset.apps/volumes-web-app-deployment-7bc5754bd4              1         1         1       10m
replicaset.apps/workflow-controller-6b9b6c5b46                     1         1         1       10m

NAME                                         READY   AGE
statefulset.apps/kserve-controller-manager   1/1     10m
statefulset.apps/metacontroller              1/1     10m
```

```
nils@NilsOld:/mnt/cs/GA$ k create -f mnist.yaml 
tfjob.kubeflow.org/tfjobd5znj created
```

```
nils@NilsOld:/mnt/cs/GA$ k get tfjob
NAME         STATE     AGE
tfjobd5znj   Created   62s
```








