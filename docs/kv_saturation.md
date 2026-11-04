# KV Saturation Flow Control

`kv_saturation` admits a request only to a replica whose KV cache has room for it, so engines are never asked to hold more context than fits and preempt running requests to make space. It runs in the Ray Serve router.

## How it works

1. **Budget.** Each replica's budget is `budget_fraction` of its KV cache capacity in tokens, minus what admitted, unfinished requests have reserved. Capacity comes from the engine itself: vLLM's KV cache block count times its block size. Engines that report no capacity, such as SGLang, make admission fail with an error instead of queueing requests forever. Lower `budget_fraction` below 1 when something outside the reservations holds KV, such as a KV connector that keeps a finished request's blocks while it saves them.
2. **Request size.** A request needs room for its prompt tokens plus all the tokens it will generate. The prompt is known when the request arrives, but its output length is not, so the plugin predicts it as the trajectory's average output so far. A trajectory's first turn has no history, so it assumes `default_osl` output tokens. The prediction is not exact, and a turn much longer than the ones before it can outgrow its reservation. Setting `budget_fraction` a little below 1.0 leaves headroom for those turns, and a turn that outgrows even that makes the engine preempt, as it would without flow control. A request larger than a whole replica reserves that replica's full capacity, so it runs alone on an idle replica.
3. **Last replica.** The replica that served the trajectory's last turn still holds most of its context, so when it has room it is the only replica allowed. Otherwise every replica with room is allowed. Picking among the allowed replicas is left to the scheduler, which scores them with the profile's scorers.
4. **Queueing.** When no replica fits, the request waits until a reservation is freed. A caller that keeps its own queue can compare each request's `kv_tokens_needed()` with each replica's `max_admissible_kv_tokens()`, so requests that still fit nowhere cost a size check instead of a full placement.
5. **Release.** When a request finishes, its reservation is freed. When the integration reports its output length, the length updates the trajectory's average and the replica becomes the one its next turn is offered first. Ray Serve doesn't report output lengths, so every Ray Serve request assumes `default_osl`.

## Configuration

```yaml
profiles:
  default:
    flow_control:
      type: kv_saturation
      default_osl: 1024   # output tokens assumed for a trajectory's first turn
      budget_fraction: 1.0  # share of each replica's KV capacity that may be reserved, in (0, 1]
    scorers:
      - type: least_queue
        weight: 1.0
    picker:
      type: max_score
```
