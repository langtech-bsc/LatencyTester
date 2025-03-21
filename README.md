# Distributed Benchmarker

## Overview
Distributed Benchmarker is designed to measure the performance and latency of APIs, code functions, or models in a distributed computing environment. Using `torchrun`, it enables multiple processes to execute tests in parallel, collect execution times

## Command Usage
```sh
torchrun --nproc_per_node=5 -m runner \
    --test-method OpenAIChatStream \
    --method-args='api_url=your-url,api_key=your-key,model=model-name'
```

### How This Works:
1. **Only Rank 0** performs a setup operation.
2. **All other ranks wait** at `torch.distributed.barrier()` until rank 0 reaches this point.
3. **Once rank 0 reaches `barrier()`**, all processes proceed together.

## Adding a New Plugin
To add a new plugin, follow these steps:

### Example Plugin Implementation
Create a new plugin by defining a class and registering it with `MethodManager`:

```python
from openai import OpenAI
from runner.methods.methods_manager import MethodManager, BaseTest

@MethodManager.register_method("MyNewPlugin")
class MyNewPlugin(BaseTest):
    def __init__(self, arg1, arg2, ...):
        self.arg1 = arg1
        self.arg1 = arg2
        ...

    def invoke(self):
        @your code here
        pass
```

### Using the New Plugin
Now, you can use your new plugin in the `torchrun` command:
```sh
torchrun --nproc_per_node=5 -m runner \
    ----method-plugin path/to/your-new-pluging.py
    --test-method MyNewPlugin \
    --method-args='Arguments for the method should be separated by commas'
```