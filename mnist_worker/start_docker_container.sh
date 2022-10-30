sudo docker run -v /mnt/kubeflow/kubeflow-mnist-pv-pvc-4ac0b9a5-3a4d-485b-a9de-024e1d99c9c3:/emnist_data 192.168.3.122:30002/mnist/mnist_worker:latest python3 mnist_worker.py /emnist_data
