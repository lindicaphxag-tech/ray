"""Benchmark epoch-lease tax using actual Ray 2.59 Learner.update steps.

Modes are raw (unsafe comparison), one remote authority ticket per update,
and one ticket per eight native updates. Each arm restarts from the same
authoritative model state and processes the same CartPole offline batch.
This measures local CPU overhead, not actual PPO reward or GPU throughput.
"""
import json
import pathlib
import sys
import ray
from rllib_two_actor_epoch_fenced_recovery import EpochAuthority, FencedNativeLearner

def main():
    source=str(pathlib.Path(sys.argv[1]).resolve())
    ray.init(num_cpus=3,include_dashboard=False,ignore_reinit_error=True,
             log_to_driver=False)
    auth=EpochAuthority.remote()
    native=FencedNativeLearner.remote(source,0,auth)
    try:
        original=ray.get(native.initial_snapshot.remote())
        ray.get(auth.bootstrap.remote(1))
        ray.get(native.restore.remote(original,1))
        ray.get(native.resume_recovered.remote(1))
        ray.get(auth.release_recovered.remote(1))
        result=ray.get(native.benchmark_native_lease_cost.remote(original,blocks=4,steps=16))
        print("NATIVE_LEASE_BENCHMARK",json.dumps(result,sort_keys=True),flush=True)
        print("BENCHMARK_VALIDATION_PASSED",flush=True)
    finally:
        ray.get(native.close.remote())
        ray.shutdown()

if __name__=="__main__":
    main()
