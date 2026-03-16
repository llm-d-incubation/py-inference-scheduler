# py-inference-scheduler
A Python implementation of an inference request scheduler, while the focus of this repo will be related to RL sampling improvements, much of what is done here will be translatable to the other inference scheduling uses & systems.

***Note***: RL is an active area of research, much of our understanding may shift over time. Expect rapid updates as we explore optimizations  

## WIP
This repo is a wip, and currently only implements the scheduling portion of the solutions described in: https://docs.google.com/document/d/1FsmXdVILBKlr9c7ZJqH-gkcR-YNu4CT6fwMQ5s9Gbsk/edit?pli=1&tab=t.0

A rough-in roadmap is [here](#roadmap).


## Ray Serve example
This example uses a simple implementation of prefix cache aware routing. This example uses Ray Serve's custom routing: https://docs.ray.io/en/latest/serve/advanced-guides/custom-request-router.html. 

### Prereqs
- Have a k8s clusters with gpu accelerators (the example uses H100s, which is admittedly overkill for this specific example, use the accelerators that best fit your scenario)
- Ensure your k8s cluster has kuberay installed
- This repo cloned
- A CLI that has ray installed (a venv was used in development)

### Steps

1. CD into this repo `cd ./some-path/py-inference-scheduler`
1. Create the scheduler configuration: `kubectl apply -f ./configs/scheduler_config.yaml`
1. Create the Ray Cluster vis RayService CRD `kubectl apply -f ./configs/ray_service.yaml`. Wait for the resources to reach `Ready`
1. Run `kubectl get svc`, copy the `ray-serve-llm-XXXXX-head-svc` service name.
1. Create a new CLI window and run `kubectl port-forward ray-serve-llm-XXXXX-head-svc 8265:8265`. This is the dashboard port, which allow our Ray CLI to communicate with this ray cluster. You can now see the dashboard at http://localhost:8265.
1. Run `ray job submit   --address="http://127.0.0.1:8265"   --working-dir="./" -- python ./integration/rayserve/router.py`. Which deploys our llm with our custom logic. Refer to the python file for more details.
1. Go to the Serve tab of the dashboard, (it may take a few minutes for the deployment to stabilize) eventually should look something like: ![alt text](./images/dash.png)

1. We need another CLI window to port forward to the service port: `kubectl port-forward svc/ray-serve-llm-XXXXX-head-svc 8000`

1. Curl your deployment multiple times to prove that the routing is irritatingly sticky! Example curl command:
```
curl \
  --location 'http://localhost:8000/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --data '{
      "model": "qwen-0.5b",
      "messages": [
          {
              "role": "system", 
              "content": "You are a helpful assistant."
          },
          {
              "role": "user", 
              "content": "Provide steps to serve an LLM using Ray Serve."
          }
      ]
  }'
```

1. To prove that all requests sharing prefixes route to the same replica, we will take a look at the OpenAiIngress logs. Ensure that you use the same configuration shown here (notw that the replicas should be the same for this example): 

![alt text](./images/scheduler_logs.png)


## Roadmap

- Implement Batch composition mechanism
- Implement RL-focused flow control to help solve
  - Sample tail latency reduction
  - Throughput improvements
  - Rebalancing samplers when long tail requests causes sampler imbalance
- Natively support sophisticated serving mechanisms such as Wide-EP with Wide-EP specific scheduling improvements
- Multi-turn/Agentic RL specific optimizations
- Experimentation with novel ideas to prove sampling efficiency improvements do not negatively impact model convergence